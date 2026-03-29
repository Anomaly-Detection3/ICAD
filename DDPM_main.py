"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""
import os
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from dataset_trend_text_pattern import detect_trend_with_entropy
from transformers import BertTokenizer, BertModel
import argparse, logging, copy
from types import SimpleNamespace
from torch.utils.data import Dataset as TDataset
from torch import optim
import torch.nn as nn
from dataset_load_no_text import KalmanFilterSSM, Dataset
from fastprogress import progress_bar
import wandb
from utils import *
from modules import UNet_conditional, EMA
from torch.utils.data import DataLoader, TensorDataset
import warnings
from eval import optimal_judgment, pot_eval
from GPU_system_monitoring import monitor_gpu_memory
import threading
from eval_methods import get_adjusted_composite_metrics
warnings.filterwarnings("ignore")

config = SimpleNamespace(
    run_name="DDPM_conditional",
    epochs=200,
    noise_steps=100,
    dataname='Swat',
    seed_train=42,
    seed_test=14,
    input_size=1,
    batch_size=64,
    img_size=51,
    hidden_size=32,
    device="cuda",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=1,
    num_workers=4,
    window_size=50,
    flag='train',
    lr=1e-3)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def M_calculate(data):

    data_mean = data.mean(axis=1)
    data_max = data.max(axis=1)
    data_min = data.min(axis=1)
    data_R = data_max - data_min
    result = data_mean/data_R

    return result



class LogicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LogicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=config.img_size,
                 window_size=config.window_size, c_in=1,
                 device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.time_dim = 256

        self.model = UNet_conditional(config.input_size, config.hidden_size, window_size, img_size, self.time_dim).to(
            device)
        self.logic_net = LogicNet(input_dim=config.img_size, hidden_dim=32).to(device)

        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def logic_relation_loss(self, logic_net, window1, window2):

        positive_sample = window1
        negative_sample = torch.roll(window2, shifts=3, dims=0)
        # negative_sample = -window2


        relation_pos = logic_net(positive_sample, window2)
        relation_neg = logic_net(positive_sample, negative_sample)


        target_pos = torch.ones_like(relation_pos)
        target_neg = torch.zeros_like(relation_neg)


        loss_fn = nn.BCELoss()
        relation_loss = loss_fn(relation_pos, target_pos) + loss_fn(relation_neg, target_neg)

        return relation_loss

    @torch.inference_mode()
    def sample(self, model, condition, cfg_scale=0):

        n = condition.shape[0]
        logging.info(f"Sampling {n} new images....")
        model.eval()
        image = condition.to(self.device)
        with torch.inference_mode():

            x = torch.randn((n, self.c_in, config.window_size, self.img_size)).to(self.device)
            # shape(16, 1, 40, 430)
            image = image.cuda()

            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, image)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):

        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)

        for images_batch in pbar:
            batch_input, batch_mean, batch_condition = images_batch
            batch_input_new = batch_input.unsqueeze(1)

            batch_input = batch_mean.to(self.device) #(4, 40, 430)
            batch_condition = batch_condition.to(self.device) #(4, 256)

            cross_loss = 0
            for i in range(len(batch_input) - 1):
                window1 = batch_input[i]
                window2 = batch_input[i + 1]
                cross_loss += self.logic_relation_loss(self.logic_net, window1, window2)

            print(f"[loss]{cross_loss.item():2.3f}")

            # print('batch', batch_input.shape, batch_condition.shape) #[6, 1, 15, 430]
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = batch_input_new.to(self.device) #(64, 1, 15, 430)
                # batch_condition (6, 256)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                # if np.random.random() < 0.1:
                #     labels = None
                predicted_noise = self.model(x_t, t, batch_condition.detach())
                loss = self.mse(noise, predicted_noise)
                print(f"[loss]{loss.item():2.3f}")


            model_loss = loss + cross_loss


            if train:
                self.train_step(model_loss)
                # wandb.log({"train_mse": loss.item(),
                #            "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={model_loss.item():2.3f}"
            end_train = time.time()
            print("Training duration", end_train - strat_train)
        return model_loss.mean().item()

    def log_images(self, model):

        test_state, test_text = self.load_pre_data(mode='test', win_size=config.window_size, data_name=config.dataname)
        print("======test load=======", test_state.shape, test_text.shape)

        test_dataset = TensorDataset(test_state, test_text)

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            drop_last=True
        )

        start_test = time.time()

        image = progress_bar(test_dataloader, leave=False)
        all_data = []
        gt_data = []

        for ele in image:
            batch_input_test, batch_text_test = ele
            ema_sampled_image = self.sample(model, batch_text_test, cfg_scale=0)

            gt_data.append(batch_input_test)
            all_data.append(ema_sampled_image.detach().cpu())

        all_data = torch.cat(all_data, dim=0)
        all_data = rearrange(all_data, 'b c h w -> (b c h) w')
        print(all_data.shape)

        gt_data = torch.cat(gt_data, dim=0)
        gt_data = gt_data.unsqueeze(1)
        gt_data = rearrange(gt_data, 'b c h w -> (b c h) w')
        end_test = time.time()
        print(end_test - start_test)
        print(all_data.shape)

        # from sklearn.preprocessing import StandardScaler

        # combined_data = np.vstack([gt_data, all_data])
        # tsne = TSNE(n_components=2, random_state=23, perplexity=23, n_iter=1000)
        # reduce_data = tsne.fit_transform(combined_data)
        #
        # reduce_original = reduce_data[:len(gt_data)]
        # reduce_enhance = reduce_data[len(gt_data):]
        #
        # plt.figure(figsize=(8, 8))
        # plt.scatter(reduce_original[:, 0], reduce_original[:, 1], c='cyan', alpha=0.6, label="Original Data",
        #             marker='o', s=50)
        # plt.scatter(reduce_enhance[:, 0], reduce_enhance[:, 1], c='purple', alpha=0.6, label="Enhanced Data",
        #             marker='x', s=50)
        # plt.title("t-SNE Result")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # # plt.grid(True)
        # plt.show()


        diff_y = abs(all_data - gt_data)
        M_test = np.array(diff_y)
        M_test = M_calculate(M_test)

        if config.dataname== 'EO':
            label_r = pd.read_csv("../EO_label_test.csv", encoding='gbk')
        elif config.dataname== 'Swat':
            label_r = pd.read_csv("../test_attach_label.csv", encoding='gbk')
        elif config.dataname== 'WADI':
            label_r = pd.read_csv("../test_label.csv", encoding='gbk')
        elif config.dataname == 'PSM':
            label_r = pd.read_csv("../test_label.csv", encoding='gbk')
        else:
            label_r = pd.read_csv("../test_label.csv", encoding='gbk')

        split = M_test.shape[0]
        y_test = label_r['label'][0: split]

        start_th, end_th, step_th = 0, 0.5, 0.0001
        result = optimal_judgment(M_test, y_test, start_th, end_th, step_th)
        print(result)

        y_test = y_test.to_numpy()
        auroc, ap, _, _, _, _, _ = get_adjusted_composite_metrics(M_test, y_test)
        print("ACU","AP",auroc, ap)

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):

        torch.save(self.model.state_dict(), os.path.join("models", run_name, config.dataname+f"anewckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, config.dataname+f"anewema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, config.dataname+f"anewoptim.pt"))

    def load_bert_model(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        return tokenizer, bert_model

    def load_pre_data(self, mode, win_size, data_name):

        dataset = Dataset(mode, win_size, data_name)
        ssm_model = KalmanFilterSSM(input_dim=config.img_size, state_dim=config.img_size)

        state_values = []
        text_all = []
        for window in dataset:
            # window=(15, 430)
            window_new = window
            trend, combined_entropy = detect_trend_with_entropy(window_new)
            state_means = ssm_model.filter(torch.tensor(window, dtype=torch.float32))
            state_values.append(state_means)
            prompt = f"The joint trend pattern of time series is {trend}, Information entropy is {combined_entropy:.4f},please describe this trend and its significance."
            text_all.append(prompt)

        state_values = torch.stack(state_values)


        def load_bert_model(model_name):
            tokenizer = BertTokenizer.from_pretrained(model_name)
            bert_model = BertModel.from_pretrained(model_name)
            return tokenizer, bert_model

        model_name = '../bert-base-uncased/tt'
        tokenizer, bert_model = load_bert_model(model_name)
        tokens = tokenizer(text_all, return_tensors='pt', padding=True, truncation=True, max_length=500)
        tokens = {key: val for key, val in tokens.items()}
        with torch.no_grad():
            text_embeddings = bert_model(**tokens)

            last_hidden_states = text_embeddings.last_hidden_state  # torch.Size([136, 487, 768])

            hidden_len = 256
            linera_layer = nn.Linear(last_hidden_states.shape[-1], hidden_len)
            adjust_output = linera_layer(last_hidden_states).mean(dim=1)


        return state_values, adjust_output

    def load_pre_data_state(self, mode, win_size, data_name):

        dataset = Dataset(mode, win_size, data_name)
        ssm_model = KalmanFilterSSM(input_dim=config.img_size, state_dim=config.img_size)

        state_values = []
        for window in dataset:
            # window=(15, 430)
            state_means = ssm_model.filter(torch.tensor(window, dtype=torch.float32))
            state_values.append(state_means.mean(dim=0))

        state_values = torch.stack(state_values)


        return state_values

    def prepare(self, args):
        mk_folders(args.run_name)

        train_state, train_text = self.load_pre_data(mode='train', win_size=config.window_size, data_name=config.dataname)
        train_state_mean = self.load_pre_data_state(mode='train', win_size=config.window_size, data_name=config.dataname)
        #(9, 50, 51) (9, 256) (9, 51)
        # test_state, test_text = self.load_pre_data(mode='test', win_size=config.window_size, data_name=config.dataname)
        # print("----------", train_state.shape, train_text.shape)

        train_dataset = TensorDataset(train_state, train_state_mean, train_text)
        # test_dataset = TensorDataset(test_state, test_text)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=True
        )

        # test_dataloader = DataLoader(
        #     dataset=test_dataset,
        #     batch_size=config.batch_size,
        #     pin_memory=True,
        #     shuffle=False,
        #     num_workers=config.num_workers,
        #     drop_last=True
        # )

        self.train_dataloader = train_dataloader
        #self.train_dataloader, self.val_dataloader = train_dataloader, test_dataloader
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.logic_net.parameters()), lr=args.lr, eps=1e-5, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)


        self.save_model(run_name=args.run_name, epoch=epoch)

    def predict(self):
        model = self.model
        file_model = torch.load("../models/DDPM_conditional/"+config.dataname+f"newckpt.pt")
        model.load_state_dict(file_model)
        self.log_images(model)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--dataname', type=str, default=config.dataname, help='data name')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')

    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

if __name__ == '__main__':
    parse_args(config)
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size)

    # --- GPU Monitoring integration ---
    stop_monitoring_event = threading.Event()

    #    - duration_minutes
    #    - experiment_name
    monitor_thread = threading.Thread(
        target=monitor_gpu_memory,
        kwargs={
            'duration_minutes': 30,
            'interval_seconds': 6,
            'experiment_name': f"Diffusion Model {config.flag.capitalize()}",
            'stop_event': stop_monitoring_event
        }
    )
    monitor_thread.start()

    try:
        start = time.time()
        if config.flag == 'train':
            set_seed(config.seed_train)
            diffuser.prepare(config)
            diffuser.fit(config)
            task_type = "Training"
        else:
            set_seed(config.seed_test)
            diffuser.predict()
            task_type = "Reasoning"
        end = time.time()

    finally:
        print("Main task finished. Stopping GPU monitor...")
        stop_monitoring_event.set()
        monitor_thread.join()
        print("GPU monitoring thread has finished.")



# if __name__ == '__main__':
#     parse_args(config)
#     diffuser = Diffusion(config.noise_steps, img_size=config.img_size)
#     if config.flag == 'train':
#         set_seed(config.seed_train)
#         start = time.time()
#         # with wandb.init(project="train_sd1", group="train2", config=config):
#         diffuser.prepare(config)
#         diffuser.fit(config)
#         end = time.time()
#
#     else:
#         start = time.time()
#         set_seed(config.seed_test)
#         diffuser.predict()
#         end = time.time()
#         print("=========end==========")