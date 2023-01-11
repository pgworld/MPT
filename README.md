# SSDPipe - MPT

Finding the optimal point to partition a model is challenging as every DNN model has different architectures. The performance of hardware compo- nents that constitute SSDPipe also affect deciding the best cutting point. For easy deployment of SSDPipe, we develop a model partitioning tool (MPT). MPT takes three steps to find the optimal partitioning point.

1. **Analysis of an input DNN model.** 

A user inputs performance numbers of major hardware components to MPT. Currently, the following three are provided as inputs to MPT: (i) floating point operations per second (FLOPS) of edge and host GPUs, (ii) the network bandwidth between Pipe-SSD and TL-Server, (iii) the number of Pipe-SSDs that participates in training. Given a DNN model to partition, MPT extracts the number of floating point operations required to execute each layer. By dividing these numbers by FLOPS of GPU, MPT can estimate the execution time (seconds) of each layer by edge and host GPUs, respectively. In addition, MPT estimates output data size of each layer.

2. **Finding the optimal cutting point.**

The next step is to estimate the training time of various partitioned models and select the best one. We partition the model by moving a cutting point from the leftmost (input) to the rightmost (output) layer and estimate the aggregate throughput of a group of PipeSSDs and the throughput of TL-Server, respectively. Then, we predict the training time of SSDPipe by including the data transfer time over the network. Finally, MTP chooses the one that shows the shortest training time.

3. **Deployment of partitioned models.**

Once the optimal point to partition is decided, MTP splits the input DNN model, generating two separate models, each of which is run on PipeSSDs and TL-Server, respectively.

## How to Use
```
python MPT.py --num_ssd={Number of SSD} --ssd_flops={FLOPS of SSD} --tl_flops={FLOPS of TL-Server} --run_num={RunNum} --model={model path} --network={network bandwith}
```
