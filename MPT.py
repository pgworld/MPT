import math, argparse
from torchstat import ModelStat
from utils import load_model_statedict, element_prod
from torchvision.models import resnet50
class MPT:

    def __init__(self, args, model_arch, port=25258):
        
        if args.model:
            self.Model = load_model_statedict(model_arch, args.model)
        else:
            self.Model = model_arch

        self.TLFlops  = args.tl_flops
        self.SSDFlops = args.ssd_flops
        self.NumSSD   = args.num_ssd
        self.RunNum   = args.run_num
        self.Network  = args.network

    def _cal_training_time(self, trans_size, flops_list):

        if element_prod(trans_size) == 0:
            return math.inf

        ssdside = sum(flops_list[0])/(self.SSDFlops*self.NumSSD)
        tlside  = max(sum(flops_list[1])/self.TLFlops, element_prod(trans_size)*4/self.Network)
        
        if ssdside > tlside:
            return ssdside+tlside/self.RunNum
        else:
            return tlside+ssdside/self.RunNum
    
    def _get_model_spec(self):

        collected_nodes = ModelStat(self.Model, (3, 224, 224), 1)._analyze_model()

        name, input_shape, flops = list(), list(), list()
        for node in collected_nodes:
            name.append(node.name)
            input_shape.append(node.input_shape)
            flops.append(node.Flops)

        return name, input_shape, flops


    def get_cutting_point(self):

        name, input_shape, flops = self._get_model_spec()
        a = math.inf
        cut_point = ''
        trainable = list()

        for names, m in self.Model.named_modules():

            if len(list(m.children())) == 0:
                trn = False
                for n, param in m.named_parameters():
                    trn = trn or param.requires_grad
                trainable.append(trn)

        for i in range(len(trainable)):
            if True in trainable[:i]:
                continue
            training_time = self._cal_training_time(input_shape[i], [flops[:i], flops[i:]])
            if training_time <= a:
                a = training_time
                cut_point = name[i]

        return cut_point
             
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'SSDPipe - Model Partitioning Tool')
    parser.add_argument('--num_ssd', default=20, type=int, help='number of PipeSSD')
    parser.add_argument('--tl_flops', default=35.58*10**12, type=float, help='flops of TL-Server')
    parser.add_argument('--ssd_flops', default=5.5*10**12, type=float, help='flops of a PipeSSD')
    parser.add_argument('--run_num', default=3, type=int, help='number of runs')
    parser.add_argument('--model', default=None, type=str, help='path to partition model')
    parser.add_argument('--network', default=1.25e+9*0.85, type=float, help='network bandwidth')

    args = parser.parse_args()

    ######################
    # e.g., ResNet 50
    ######################
    model = resnet50(pretrained=False)

    for name, m in model.named_modules():
        if 'fc' in name:
            continue
        if len(list(m.children())) == 0:
            for n, param in m.named_parameters():
                param.requires_grad = False

    mpt = MPT(args, model)

    print('='*40)
    print(' '*14+'Arguments')
    for arg in sorted(vars(args)):
        print(f'{arg}:{getattr(args, arg)}')
    print('='*40)

    print(f'From "{mpt.get_cutting_point()}" layer it starts on TL-Server')
