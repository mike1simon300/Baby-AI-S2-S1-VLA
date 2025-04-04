import argparse
import contextlib
import datetime
import time
from tqdm import tqdm
import yaml
import torch_ac
import tensorboardX
import os
import sys
from model import ACModel, SmallTransformerACModel
import utils


class ConfigParser:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_model_dir(self, model_name=None):
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        training = self.get("training")
        model_name = model_name or training.get("model_name") or f"{self.get('env')['name']}_{training['algo']}_seed{training['seed']}_{date}"
        return utils.get_model_dir(model_name)

    def get_loggers(self, model_dir):
        txt_logger = utils.get_txt_logger(model_dir)
        csv_file, csv_logger = utils.get_csv_logger(model_dir)
        tb_writer = tensorboardX.SummaryWriter(model_dir)
        return txt_logger, csv_file, csv_logger, tb_writer

    def initialize_logging(self):
        model_dir = self.get_model_dir()
        return model_dir, self.get_loggers(model_dir)



class Trainer:
    def __init__(self, config_path="config.yaml"):
        self.config = ConfigParser(config_path)
        self.device = utils.device
        self.model_config = self.config.get("model")
        self.training_config = self.config.get("training")
        self.env_config = self.config.get("env")
        self._set_seed()
        self._initialize_logging()
        self._initialize_envs()
        self.model = self.initialize_model(self.obs_space, self.envs[0].action_space, self.device)
        self.algo = self.initialize_algorithm(self.envs, self.model, self.device, self.preprocess_obss)

    def get_obs_space(self, env):
        return utils.get_obss_preprocessor(env.observation_space)

    def _set_seed(self):
        utils.seed(self.config.get("training")["seed"])

    def _initialize_logging(self):
        self.model_dir, (self.txt_logger, self.csv_file, self.csv_logger, self.tb_writer) = self.config.initialize_logging()

    def get_envs(self):
        # Environments remain initialized here in the trainer.
        
        seed = self.training_config["seed"]
        procs = self.env_config["procs"]
        print(self.env_config["name"])
        return [utils.make_env(self.env_config["name"], seed + 10000 * i) for i in range(procs)]

    def _initialize_envs(self):
        self.envs = self.get_envs()
        self.obs_space, self.preprocess_obss = self.get_obs_space(self.envs[0])



    def train(self):
        num_frames, update = 0, 0
        start_time = time.time()
        total_frames = self.training_config["frames"]
        log_interval = self.training_config["log_interval"]
        save_interval = self.training_config["save_interval"]
        
        with tqdm(total=total_frames, unit='frames', desc='Training Progress') as pbar:
            while num_frames < total_frames:
                exps, logs1 = self.algo.collect_experiences()
                logs2 = self.algo.update_parameters(exps)
                logs = {**logs1, **logs2}
                num_frames += logs["num_frames"]
                update += 1
                
                pbar.update(logs["num_frames"])  # Update progress bar
                if update % log_interval == 0:
                    fps = logs["num_frames"] / (time.time() - start_time)
                    rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
                    mean_reward = list(rreturn_per_episode.values())[0]
                    nF = list(num_frames_per_episode.values())
                    if self.training_config["log_loss"]:
                        text = f"H {logs['entropy']:.3f} | V {logs['value']:.3f}"
                        text += f" | pL {logs['policy_loss']:.3f} | vL {logs['value_loss']:.3f}"
                        text += f" | ∇ {logs['grad_norm']:.3f}"
                        tqdm.write(text)
                    pbar.set_postfix({"fps": fps, "mR": mean_reward, "F:μσmM": f"{nF[0]:.1f} {nF[1]:.1f} {nF[2]} {nF[3]}"})
                if save_interval > 0 and update % save_interval == 0:
                    status = {
                        "num_frames": num_frames, 
                        "update": update,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.algo.optimizer.state_dict()
                    }
                    utils.save_status(status, self.model_dir)
                    with open(os.path.join(self.model_dir, "config.yaml"), "w") as file:
                        yaml.dump(self.config, file)
                    tqdm.write("Status saved")

    def _log_training(self, update, num_frames, start_time, logs):
        fps = logs["num_frames"] / (time.time() - start_time)
        duration = int(time.time() - start_time)
        self.txt_logger.info(f"Update {update} | Frames {num_frames} | FPS {fps} | Duration {duration}")

    def initialize_model(self, obs_space, action_space, device):
        acmodel = SmallTransformerACModel(
            obs_space, 
            action_space,
            patch_size=self.model_config["patch_size"],
            num_layers=self.model_config["num_layers"],
            num_heads=self.model_config["num_heads"],
            embed_dim=self.model_config["embed_dim"]
        )
        acmodel.to(device)
        return acmodel

    def initialize_algorithm(self, envs, acmodel, device, preprocess_obss):
        algo_name = self.training_config["algo"]
        # Note: Here we pass the entire training config. In a real case, you might want to filter out
        # only parameters relevant to the algorithm initializer.
        if algo_name == "a2c":
            algo = torch_ac.A2CAlgo(envs, acmodel, device, self.training_config["frames_per_proc"], 
                                    self.training_config["discount"], self.training_config["lr"], self.training_config["gae_lambda"],
                                    self.training_config["entropy_coef"], self.training_config["value_loss_coef"], 
                                    self.training_config["max_grad_norm"], self.training_config["recurrence"],
                                    self.training_config["optim_alpha"], self.training_config["optim_eps"], preprocess_obss)
        elif algo_name == "ppo":
            algo = torch_ac.PPOAlgo(envs, acmodel, device, self.training_config["frames_per_proc"], 
                                    self.training_config["discount"], self.training_config["lr"], self.training_config["gae_lambda"],
                                    self.training_config["entropy_coef"], self.training_config["value_loss_coef"], 
                                    self.training_config["max_grad_norm"], self.training_config["recurrence"], self.training_config["optim_alpha"], 
                                    self.training_config["clip_eps"], self.training_config["epochs"], 
                                    self.training_config["batch_size"], preprocess_obss)
        else:
            raise ValueError(f"Invalid algorithm name: {algo_name}")
        return algo

if __name__ == "__main__":
    config_file = "../configs/RL_training_config.yaml"
    trainer = Trainer(config_file)
    trainer.train()