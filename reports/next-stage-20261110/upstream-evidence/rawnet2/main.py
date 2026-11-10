import argparse
import sys
import os
from data_utils import Dataset_commom_raw,Dataset_commom_np
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import torch
from torch import nn
from model.RawGAT_ST import RawGAT_ST 
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from config_dict import MODEL_DIC
from tqdm import tqdm
from mertics import compute_eer

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def evaluate_accuracy(data_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for idx, (batch_x, batch_y) in enumerate(tqdm(data_loader,desc="dev Evaluating")):
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        batch_out = model(batch_x,Freq_aug=False)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    key_list = []    
    score_list = []

    for idx, (batch_x, batch_y) in enumerate(tqdm(data_loader,desc="Evaluating")):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,Freq_aug=False)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        # add outputs
        key_list.extend(
          ['bonafide' if key == 1 else 'spoof' for key in list(batch_y)])
        score_list.extend(batch_score.tolist())
        
    with open(save_path, 'w') as fh:
        for k, cm in zip(key_list, score_list):
            fh.write('{} {} {} {}\n'.format(k, cm))
    print('Result saved to {}'.format(save_path))

def train_epoch(data_loader, model, optimizer, device, criterion):
    running_loss = 0
    num_total = 0.0
    model.train()
    # set objective (Loss) functions --> WCE
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    
    for idx, (batch_x, batch_y ) in enumerate(tqdm(data_loader,desc="Training")):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.type(torch.int64).to(device)
        batch_out = model(batch_x,Freq_aug=True)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    return running_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 RawGAT-ST model')
    
    # Dataset
    parser.add_argument('--train_datafile', type=str, default='${EXPERT_ROOT}/detectionLib/database/ASVspoof2019train_label.txt', help='Change this to user\'s full directory address of LA database,train file')
    parser.add_argument('--dev_datafile', type=str, default='${EXPERT_ROOT}/detectionLib/database/ASVspoof2019dev_label.txt', help='Change this to user\'s full directory address of LA database,dev file')
    # Hyperparameters
    parser.add_argument('-g','--gpu', type=str, help="GPU index", default="0")
    parser.add_argument('-b','--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE',help='Weighted Cross Entropy Loss ')

    # model
    parser.add_argument('--config', type=str,
                        default='config/RawNet2.yaml', help='Model config yaml')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--features', type=str, default='Raw_GAT')

    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    
    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    dir_yaml = args.config
    with open(dir_yaml, 'r') as f_yaml:
        model_parser = yaml.load(f_yaml,Loader=yaml.FullLoader)


    #make experiment reproducible
    set_random_seed(args.seed, args)
     # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        model_parser['model_name'], args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
     
    # transforms = transforms.Compose([
    #     lambda x: pad(x),
    #     lambda x: Tensor(x)   
    # ])

    # #GPU device
    # check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #model 
    model = MODEL_DIC[model_parser['model_name']](model_parser['model'])
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    # # Inference
    # if args.eval:
    #     assert args.eval_output is not None, 'You must provide an output path'
    #     assert args.model_path is not None, 'You must provide model checkpoint'
    #     produce_evaluation_file(dev_set, model, device, args.eval_output)
    #     sys.exit(0)

    # Training Dataloader
    train_set = Dataset_commom_raw(data_file=args.train_datafile)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # # validation Dataloader
    dev_set = Dataset_commom_raw(data_file=args.dev_datafile)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        running_loss = train_epoch(train_loader,model, optimizer, device, criterion)
        writer.add_scalar('loss', running_loss, epoch)
        if(dev_loader):
            val_loss = evaluate_accuracy(dev_loader, model, device)
            writer.add_scalar('val_loss', val_loss, epoch)
        print('\n{} - {} '.format(epoch, running_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}_{}.pth'.format(epoch,running_loss)))
