import pickle
from pathlib import Path
from time import time

import gpytorch
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt

import src


def get_map(cfg):
    with np.load(f"./arrays/{cfg.map.geo_coordinate}.npz") as data:
        map = data["arr_0"]
    print(f"Loaded map with shape {map.shape}.")
    return map


def get_sensor(cfg, map, rng):
    rate = cfg.sensor.sensing_rate
    noise_scale = cfg.sensor.noise_scale
    sensor = src.sensors.PointSensor(
        matrix=map, # map elevation, numpy array shape(W, H)
        env_extent=cfg.map.env_extent, # map range
        rate=rate, # sensor working rate
        noise_scale=noise_scale, # noise scale
        rng=rng,
    )
    print(f"Initialized sensor with rate {rate} and noise scale {noise_scale}.")
    return sensor


def get_robot(cfg, sensor):
    init_state = np.array([cfg.map.task_extent[1], cfg.map.task_extent[2], -np.pi]) # init state: [x, y, theta]
    robot = src.robots.DiffDriveRobot(
        sensor=sensor,
        state=init_state,
        control_rate=cfg.robot.control_rate,
        max_lin_vel=cfg.robot.max_lin_vel,
        max_ang_vel=cfg.robot.max_ang_vel,
        goal_radius=cfg.robot.goal_radius,
    )
    print(f"Initialized robot.")
    return robot


def get_planner(cfg, rng):
    if cfg.planner.name == "max_entropy":
        planner = src.planners.MaxEntropyPlanner(
            cfg.map.task_extent, rng, cfg.planner.num_candidates
        )
    else:
        raise ValueError(f"Unknown planner: {cfg.planner.name}")
    print(f"Initialized planner {cfg.planner.name}.")
    return planner


def get_visualizer(cfg, map):
    visualizer = src.utils.Visualizer(
        cfg.map.env_extent, cfg.map.task_extent, cfg.plot_robot_interval
    )
    vmin, vmax = np.min(map), np.max(map)
    visualizer.vmins[1], visualizer.vmaxs[1] = vmin, vmax
    visualizer.plot_image(
        index=0, matrix=map, title="Ground Truth", vmin=vmin, vmax=vmax
    )
    print(f"Initialized visualizer.")
    return visualizer


def get_evaluator(cfg, sensor):
    # x_min, x_max, y_min, y_max is cfg.map.task_extent
    evaluator = src.utils.Evaluator(sensor, cfg.map.task_extent, cfg.eval_grid)
    # print(f"YCY TEST Evaluator initialized with grid size {cfg.eval_grid} and task extent {cfg.map.task_extent}.")
    print(f"Initialized evaluator.")
    return evaluator


def pilot_survey(cfg, robot, rng):
    bezier_planner = src.planners.BezierPlanner(cfg.map.task_extent, rng)
    goals = bezier_planner.plan(num_points=cfg.num_bezier_points)
    robot.goals = goals
    while len(robot.goals) > 0:
        robot.step()
    x_init, y_init = robot.commit_samples() # many points
    print(f"Collected {len(x_init)} samples in pilot survey.")
    return x_init, y_init


def get_scalers(cfg, x_init, y_init):
    x_scaler = src.scalers.MinMaxScaler()
    x_scaler.fit(x_init)
    y_scaler = src.scalers.StandardScaler()
    y_scaler.fit(y_init)
    return x_scaler, y_scaler


def get_kernel(cfg, x_init):
    if cfg.kernel.name == "rbf":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = cfg.kernel.lengthscale
        kernel.outputscale = cfg.kernel.outputscale
    elif cfg.kernel.name == "ak":
        kernel = gpytorch.kernels.ScaleKernel(
            src.kernels.AttentiveKernel(
                dim_input=x_init.shape[1],
                dim_hidden=cfg.kernel.dim_hidden,
                dim_output=cfg.kernel.dim_output,
                min_lengthscale=cfg.kernel.min_lengthscale,
                max_lengthscale=cfg.kernel.max_lengthscale,
            )
        )
    elif cfg.kernel.name == "sparse":
        # evaluator = src.utils.Evaluator(sensor, cfg.map.task_extent, cfg.eval_grid)
        print(f'YCY-DEBUG kernel.name == "sparse"')
        x_min, x_max, y_min, y_max = cfg.map.task_extent
        num_x, num_y = cfg.eval_grid
        kernel_len = torch.ones(num_x, num_y).mul(cfg.kernel.init_klen).requires_grad_(True)    # 2D length scale matrix
        kernel_scale = torch.ones(num_x, num_y).requires_grad_(True)  
        res = (x_max - x_min)/num_x # TODO: assumpation res_x = res_y

        kernel = src.kernels.CovSparseKernel2D(
            kLenMat=kernel_len,
            kScaleMat = kernel_scale,
            minX = x_min,
            minY = y_min,
            resolution = res
        )

    else:
        raise ValueError(f"Unknown kernel: {cfg.kernel}")
    print(f"Initialized kernel {cfg.kernel.name}.")
    return kernel


def get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel):
    if "ssgp" in cfg.model.name:
        model = src.models.SSGPModel(
            num_inducing=cfg.model.num_inducing,
            learn_inducing=cfg.model.learn_inducing,
            strategy_inducing=cfg.model.strategy_inducing,
            x_train=x_init,
            y_train=y_init,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            kernel=kernel,
            noise_variance=cfg.likelihood.noise_variance,
            batch_size=cfg.model.batch_size,
            jitter=cfg.model.jitter,
        )
    elif "poam" in cfg.model.name:
        model = src.models.POAMModel(
            num_inducing=cfg.model.num_inducing,
            learn_inducing=cfg.model.learn_inducing,
            x_train=x_init,
            y_train=y_init,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            kernel=kernel,
            noise_variance=cfg.likelihood.noise_variance,
            batch_size=cfg.model.batch_size,
            jitter=cfg.model.jitter,
        )
    elif "ovc" in cfg.model.name:
        model = src.models.OVCModel(
            num_inducing=cfg.model.num_inducing,
            learn_inducing=cfg.model.learn_inducing,
            x_train=x_init,
            y_train=y_init,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            kernel=kernel,
            noise_variance=cfg.likelihood.noise_variance,
            batch_size=cfg.model.batch_size,
            jitter=cfg.model.jitter,
        )
    elif "ibgki" in cfg.model.name:
        print(f'YCY-DEBUG model.name == "ibgki"')
        model = src.models.IndependentBGKIModel(
            x_train=x_init,
            y_train=y_init,
            kernel=kernel,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            jitter=cfg.model.jitter
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    print(f"Initialized model {cfg.model.name}.")
    return model


def model_update(cfg, model, evaluator):
    print("Updating model...")
    start = time()
    if "ibgki" not in cfg.model.name:
        model.update_inducing()
        model.update_variational()
    losses = model.optimize(num_steps=cfg.num_train_steps) #optim num_train_steps
    end = time()
    evaluator.training_times.append(end - start)
    evaluator.losses.extend(losses)


# def evaluation(model, evaluator):
#     print("Evaluating model...")
#     start = time()
#     mean, std = model.predict(evaluator.eval_inputs) # NOTE: evaluator.eval_inputs is the meshgrid covered allmap
#     end = time()
#     evaluator.prediction_times.append(end - start) # NOTE: compute metrics
#     evaluator.compute_metrics(mean, std)

def evaluation(model, evaluator):
    print("Evaluating model...")
    start = time()
    print(f'YCY-DEBUG, waiting to predictation through ibgki')
    mean, std = model.predict(evaluator.eval_inputs) # NOTE: evaluator.eval_inputs is the meshgrid covered allmap
    print(f'YCY-DEBUG')
    print(f'evaluator.eval_inputs.shape:{evaluator.eval_inputs.shape}')
    print(f'mean.shape:{mean.shape}')
    print(f'std.shape:{std.shape}')
    
    end = time()
    evaluator.prediction_times.append(end - start) # NOTE: compute metrics
    evaluator.compute_metrics(mean, std)


def visualization(visualizer, evaluator, x_inducing=None):
    print(f"Visualizing results...")
    visualizer.plot_prediction(evaluator.mean, evaluator.std, evaluator.abs_error)
    visualizer.plot_data(evaluator.x_train)
    if x_inducing is not None:
        visualizer.plot_inducing_inputs(x_inducing)
    visualizer.plot_metrics(evaluator)


def information_gathering(robot, model, planner, visualizer=None):
    while True:
        print("Planning...")
        goal = planner.plan(model, robot.state[:2])# MaxEntropyPlanner---> next goal point
        if visualizer is not None:
            visualizer.plot_goal(goal)
            visualizer.pause()
        robot.goals = goal
        plot_counter = 0
        print("Sampling...")
        while robot.has_goals:
            plot_counter += 1
            robot.step() # robot step and sample points
            if visualizer is None:
                continue
            if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
                visualizer.plot_robot(robot.state, scale=4)
                visualizer.pause()
        if len(robot.sampled_observations) > 0: # return sampled points
            x_new, y_new = robot.commit_samples()
            return x_new, y_new


def save_evaluator(evaluator, save_path):
    with open(save_path, "wb") as file:  # Overwrites any existing file.
        pickle.dump(evaluator, file, pickle.HIGHEST_PROTOCOL)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # NOTE: 1. print config to debug
    print(f'==========1. print config to debug, config path ./experiments/benchmark/configs/...')
    print(OmegaConf.to_yaml(cfg))

    # NOTE: 2. experiment setup
    print(f'==========2. getting experiment setip, include rng, map, sensor, robot, planner')
    rng = src.utils.set_random_seed(cfg.seed) # type: np.random.RandomState(seed) | rng:RandomState(MT19937)
    map = get_map(cfg) #numpy array shape(360, 360) 
    """
    print(map)
    [[106. 105. 108. ...  94.  95.  95.]
    [104. 101. 102. ... 121. 114. 111.]
    [102.  99.  96. ... 158. 145. 134.]
    ...
    [ 52.  54.  57. ... 129. 123. 124.]
    [ 53.  55.  57. ... 139. 129. 124.]
    [ 55.  59.  62. ... 145. 133. 130.]]
    """
    sensor = get_sensor(cfg, map, rng) # type: src.sensors.PointSensor
    robot = get_robot(cfg, sensor) # type: src.robots.DiffDriveRobot
    planner = get_planner(cfg, rng)

    # visualization and evaluation
    if cfg.visualize:
        visualizer = get_visualizer(cfg, map)
    evaluator = get_evaluator(cfg, sensor)

    # pilot survey
    x_init, y_init = pilot_survey(cfg, robot, rng)
    print(f'YCY-DEBUG x_init.shape:{x_init.shape} y_init.shape:{y_init.shape}')
    evaluator.add_data(x_init, y_init)
    x_scaler, y_scaler = get_scalers(cfg, x_init, y_init)
    kernel = get_kernel(cfg, x_init)
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel) # model  

    # First optimization takes longer time, which affects the training time evaluation.
    model.optimize(num_steps=1)# GPU warm start


    # exit()
    model_update(cfg, model, evaluator) # NOTE: optim model within num_train_steps step, and append traintime and trainLoss toevaluator  
    evaluation(model, evaluator) # NOTE: evaluate once  
    print(f'\n ==========init evaluation metrics')
    print(f"evaluator.smses[-1]: {evaluator.smses[-1]}")
    print(f"evaluator.rmses[-1]: {evaluator.rmses[-1]}")
    print(f"evaluator.maes[-1] : {evaluator.maes[-1]}")
    print(f"evaluator.mslls[-1]: {evaluator.mslls[-1]}")
    print(f"evaluator.nlpds[-1]: {evaluator.nlpds[-1]}")
    print(f"evaluator.training_times[-1]: {evaluator.training_times[-1]}")
    print(f"evaluator.prediction_times[-1]: {evaluator.prediction_times[-1]}")
    if cfg.visualize:
        visualization(visualizer, evaluator)
        print("Press any key to continue and [ESC] to exit...")
        plt.waitforbuttonpress()

    # main loop
    decision_epoch = 0
    start_time = time()
    while True: # NOTE: loop, MaxEntropyPlanner to get the next goal and sample dataset by using sensors
        num_samples = len(evaluator.y_train)
        if num_samples >= cfg.max_num_samples:# breakdown
            break
        time_elapsed = time() - start_time
        decision_epoch += 1

        print(
            f"Decision epoch: {decision_epoch} | "
            + f" Time used: {time_elapsed:.2f} seconds | "
            + f"Number of samples: {num_samples} / {cfg.max_num_samples}"
        )

        # NOTE: information gathering， MaxEntropyPlanner to get the next goal and sample dataset by using sensors
        x_new, y_new = information_gathering(
            robot, model, planner, visualizer if cfg.visualize else None
        )

        evaluator.add_data(x_new, y_new) # NOTE: add new data to evaluator  
        model.add_data(x_new, y_new) # NOTE: add new data to model
        model_update(cfg, model, evaluator)
        evaluation(model, evaluator)

        if not cfg.visualize:
            continue
        visualizer.clear()
        visualizer.plot_title(decision_epoch, time_elapsed)
        visualization(
            visualizer,
            evaluator,
            model.x_inducing if hasattr(model, "x_inducing") else model.train_x,
        )
        if cfg.kernel.name == "ak":
            visualizer.plot_lengthscales(
                model, evaluator, cfg.kernel.min_lengthscale, cfg.kernel.max_lengthscale
            )
        if cfg.kernel.name == "sparse":
            visualizer.plot_lengthscales_ibgki(
                model, 0, cfg.kernel.init_klen
            )
        visualizer.pause()
    
    print("Done!")
    print(f'\n ==========print final metrics')
    print(f"evaluator.smses[-1]: {evaluator.smses[-1]}")
    print(f"evaluator.rmses[-1]: {evaluator.rmses[-1]}")
    print(f"evaluator.maes[-1] : {evaluator.maes[-1]}")
    print(f"evaluator.mslls[-1]: {evaluator.mslls[-1]}")
    print(f"evaluator.nlpds[-1]: {evaluator.nlpds[-1]}")
    print(f"evaluator.training_times[-1]: {evaluator.training_times[-1]}")
    print(f"evaluator.prediction_times[-1]: {evaluator.prediction_times[-1]}")

    print(f'\n ==========print mean of metrics')

    print(f"np.mean(evaluator.smses): {np.mean(evaluator.smses)}")
    print(f"np.mean(evaluator.rmses): {np.mean(evaluator.rmses)}")
    print(f"np.mean(evaluator.maes ): {np.mean(evaluator.maes)}")
    print(f"np.mean(evaluator.mslls): {np.mean(evaluator.mslls)}")
    print(f"np.mean(evaluator.nlpds): {np.mean(evaluator.nlpds)}")
    print(f"np.mean(evaluator.training_times): {np.mean(evaluator.training_times)}")
    print(f"np.mean(evaluator.prediction_times): {np.mean(evaluator.prediction_times)}")
    

    if cfg.visualize:
        visualizer.show()

    # save results
    if cfg.save_results:
        if cfg.kernel.name == "ak":
            lenscale = model.get_ak_lengthscales(evaluator.eval_inputs).reshape(
                *evaluator.eval_grid
            )
            evaluator.lenscale = lenscale
        if hasattr(model, "x_inducing"):
            evaluator.x_inducing = model.x_inducing
        run_id = f"{cfg.map.geo_coordinate}_{cfg.model.name}_{cfg.seed}"
        save_path = f"{cfg.output_folder}/{run_id}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_evaluator(evaluator, f"{save_path}/evaluator.pkl")
        OmegaConf.save(cfg, f"{save_path}/config.yaml")
        print(f"Results saved to {save_path}.")


if __name__ == "__main__":
    main()
