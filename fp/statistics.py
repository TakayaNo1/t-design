import sys
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from scipy import stats
from pathlib import Path
import re
import math
from PIL import Image
import matplotlib.pyplot as plt

#link {https://www.math.wustl.edu/~sawyer/handouts/Jackknife.pdf}
#link {https://towardsdatascience.com/calculating-confidence-interval-with-bootstrapping-872c657c058d}

ROOT_PATH = "/mnt/d/quantum"

B12 = "B12"
B15 = "B15"
B12_JACKKNIFE = "B12_Jackknife"

class Para:
    def __init__(self, culcType=B12, circuit="LRC", Ntimes=1000, depth=10, t=1, Nq=7, Nsample=100) -> None:
        self.culcType=culcType
        self.circuit=circuit
        self.Ntimes=Ntimes
        self.depth=depth
        self.t=t
        self.Nq=Nq
        self.Nsample=Nsample

def get_file_name(para : Para):
    if para.culcType == B12_JACKKNIFE:
        if para.circuit=='RC':
            return f'{para.circuit}_Nq{para.Nq}_t{para.t}_sample{para.Nsample}'
        if para.circuit=='LRC' or para.circuit=='RDC':
            return f'{para.circuit}_Nq{para.Nq}_depth{para.depth}_t{para.t}_sample{para.Nsample}'
    elif para.culcType == B12 or para.culcType == B15:
        if para.circuit=='RC':
            return f'{para.circuit}_Nq{para.Nq}_{para.Ntimes}times_t{para.t}_sample{para.Nsample}'
        if para.circuit=='LRC' or para.circuit=='RDC':
            return f'{para.circuit}_Nq{para.Nq}_{para.Ntimes}times_depth{para.depth}_t{para.t}_sample{para.Nsample}'
    else:
        return None

def glob_file_name(para : Para):
    return f'{para.circuit}_Nq{para.Nq}_*times_depth{para.depth}_t{para.t}_sample*'    

def get_data(para : Para) -> Series or None:
    try:
        if para.culcType == B12_JACKKNIFE:
            return pd.read_csv(f"{ROOT_PATH}/result/jackknife/"+get_file_name(para)+".csv")['value']
        elif para.culcType == B12 or para.culcType == B15:
            return pd.read_csv(f"{ROOT_PATH}/result/{para.Ntimes}times/"+get_file_name(para)+".csv")['value']
    except FileNotFoundError:
        return None
    except KeyError:
        return None

def ecdf(data :Series):
    n=len(data)
    x=np.sort(data)
    y=np.arange(1,n+1)/n
    return x,y

def save_plot_ecdf_image(data :Series, para :Para):
    """Plot ecdf"""
    # Call get_ecdf function and assign the returning values
    x, y = ecdf(data)
    
    plt.clf()
    plt.plot(x,y,marker='.',linestyle='none')
    plt.xlabel("FP")
    plt.ylabel("")
    plt.title("ECDF")
    plt.savefig("{ROOT_PATH}/result/figure/ecdf_"+get_file_name(para)+".png")

def draw_bs_replicates(data :Series, func, size :int):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return bs_replicates

def save_some_plot_image(para:Para):
    plt.clf()
    data_files=[]
    for f in Path("{ROOT_PATH}/result/").glob(f"*/{glob_file_name(para)}.csv"):
        data_files.append(str(f))
    size=len(data_files)
    fig = plt.figure(figsize=(5*size,5))
    percentiles = {}
    for i in range(size):
        file=data_files[i]
        Ntimes=int(re.findall(r'_([0-9]+)times_', str(file))[0])
        Nsample=int(re.findall(r'_sample([0-9]+).', str(file))[0])
        para.Ntimes=Ntimes
        para.Nsample=Nsample
        data=get_data(para)
        bs_replicates_data = draw_bs_replicates(data,np.mean,15000)

        hist_ax=fig.add_subplot(2,size,i+1)
        bs_ax=fig.add_subplot(2,size,size+i+1)

        hist_ax.set_title(f"{Ntimes}times Nsample:{Nsample}")
        hist_ax.hist(data,bins=30,density=True)
        # p1=[2.5]
        # p2=[97.5]
        p1=[5.5]
        p2=[94.5]
        pt_left=np.percentile(data,p1)
        pt_center=np.percentile(data,[50])
        pt_right=np.percentile(data,p2)
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{p1[0]}']=pt_left
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{50}']=pt_center
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{p2[0]}']=pt_right
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_minus']=pt_left-pt_center
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_plus']=pt_right-pt_center
        hist_ax.axvline(x=pt_left, ymin=0, ymax=1,label=f'{p1[0]}th percentile',c='y')
        hist_ax.axvline(x=pt_right, ymin=0, ymax=1,label=f'{p2[0]}th percentile',c='r')
        # hist_ax.legend(loc = 'upper right')
        bs_ax.hist(bs_replicates_data,bins=30,density=True)
        pt_left=np.percentile(bs_replicates_data,[2.5])
        pt_center=np.percentile(bs_replicates_data,[50])
        pt_right=np.percentile(bs_replicates_data,[97.5])
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{p1[0]}_bs']=pt_left
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{50}_bs']=pt_center
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_{p2[0]}_bs']=pt_right
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_minus_bs']=pt_left-pt_center
        percentiles[f'{para.depth}_{para.t}_{Ntimes}_{Nsample}_plus_bs']=pt_right-pt_center
        bs_ax.axvline(x=pt_left, ymin=0, ymax=1,label=f'{p1[0]}th percentile',c='y')
        bs_ax.axvline(x=pt_right, ymin=0, ymax=1,label=f'{p2[0]}th percentile',c='r')
    # fig.tight_layout()              #レイアウトの設定
    for k, v in percentiles.items():
        print(k, "\t", v[0])

    plt.savefig(f"{ROOT_PATH}/result/figure/all_Nq{para.Nq}_depth{para.depth}_t{para.t}.png")

def test():
    fig = plt.figure(figsize=(5*2,5))
    data=np.random.normal(loc=1, scale=1, size=5000)
    data=np.append(data, np.random.normal(loc=5, scale=1, size=7500))
    bs_replicates_data = draw_bs_replicates(data,np.mean,5000)

    hist_ax=fig.add_subplot(2,1,1)
    bs_ax=fig.add_subplot(2,1,2)

    # hist_ax.set_title(f"{Ntimes}times Nsample:{Nsample}")
    # p1=[2.5]
    # p2=[97.5]
    p1=[5.5]
    p2=[94.5]
    hist_ax.hist(data,bins=30,density=True)
    pt_left=np.percentile(data,p1)
    pt_right=np.percentile(data,p2)
    hist_ax.axvline(x=pt_left, ymin=0, ymax=1,label=f'{p1[0]}th percentile',c='y')
    hist_ax.axvline(x=pt_right, ymin=0, ymax=1,label=f'{p2[0]}th percentile',c='r')
    bs_ax.hist(bs_replicates_data,bins=30,density=True)
    pt_left=np.percentile(bs_replicates_data,p1)
    pt_right=np.percentile(bs_replicates_data,p1)
    bs_ax.axvline(x=pt_left, ymin=0, ymax=1,label=f'{p1[0]}th percentile',c='y')
    bs_ax.axvline(x=pt_right, ymin=0, ymax=1,label=f'{p2[0]}th percentile',c='r')

    plt.savefig(f"{ROOT_PATH}/result/figure/test.png")

def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=True, paired=False, axis=0, confidence_level=0.95,
              method='BCa', random_state=None):
    # Input validation
    args = stats._resampling._bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, n_resamples, batch, method,
                         random_state)
    data, statistic, vectorized, paired, axis = args[:5]
    confidence_level, n_resamples, batch, method, random_state = args[5:]

    theta_hat_b = []

    batch_nominal = batch or n_resamples

    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = stats._resampling._bootstrap_resample(sample, n_resamples=batch_actual,
                                           random_state=random_state)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = (1 - confidence_level)/2
    if method == 'bca':
        interval = stats._resampling._bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)
        percentile_fun = stats._resampling._percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    # return stats._resampling.BootstrapResult(confidence_interval=stats._resampling.ConfidenceInterval(ci_l, ci_u),
    #                        standard_error=np.std(theta_hat_b, ddof=1, axis=-1))
    return ci_l, np.mean(theta_hat_b), ci_u

def chebyshevs_inequality_confidence_interval(data :Series, k:int=3) -> tuple[float, float]:
    N=len(data)
    mean=np.mean(data)
    std=np.std(data)
    H=k*std/math.sqrt(N)
    return mean, H

def main_jackknife():
    Nq_list=[2,3,4,5,6,7]
    t_list=[1,2,3,4]
    depth=10
    circuit_list={'RC', 'LRC', 'RDC'}
    csv=DataFrame(columns=Nq_list, index=t_list, dtype=object)
    for circuit in circuit_list:
        for Nq in Nq_list:
            for t in t_list:
                para=Para(culcType=B12_JACKKNIFE, circuit=circuit, t=t, depth=depth, Nq=Nq, Nsample=1000)
                data=get_data(para)
                if data is None:
                    continue
                ps_mean=np.mean(data)
                vps=np.sum([math.pow(psi-ps_mean, 2) for psi in data]) / (len(data)-1)
                conf = 1.960 * math.sqrt(vps/len(data))
                csv[Nq][t] = f"{round(ps_mean, 3)}{'±'}{round(conf, 3)}"
        print(csv)
        csv.to_csv(f'{ROOT_PATH}/result/total/jackknife95_{circuit}_D{depth}.csv')

    Nq_list=[2,3,4,5,6,7]
    depth_list=[3,4,5,6,7,8,9,10]
    t_list=[1,2,3,4]
    circuit_list={'LRC', 'RDC'}
    for circuit in circuit_list:
        for t in t_list:
            csv=DataFrame(columns=Nq_list, index=depth_list, dtype=object)
            for Nq in Nq_list:
                for depth in depth_list:
                    para=Para(culcType=B12_JACKKNIFE, circuit=circuit, t=t, depth=depth, Nq=Nq, Nsample=1000)
                    data=get_data(para)
                    if data is None:
                        continue
                    ps_mean=np.mean(data)
                    vps=np.sum([math.pow(psi-ps_mean, 2) for psi in data]) / (len(data)-1)
                    conf = 1.960 * math.sqrt(vps/len(data))
                    csv[Nq][depth] = f"{round(ps_mean, 3)}{'±'}{round(conf, 3)}"
            print(csv)
            csv.to_csv(f'{ROOT_PATH}/result/total/jackknife95_{circuit}_t{t}.csv')

    # Nq_list=[2,3,4,5,6,7]
    # t_list=[1,2,3,4]
    # circuit_list={'LRC', 'RDC'}
    # for circuit in circuit_list:
    #     for t in t_list:
    #         csv=DataFrame(columns=[f"{Nq}qubit" for Nq in Nq_list], index=["ps", "vps", "conf", "skew", "kurt", "result"], dtype=object)
    #         for Nq in Nq_list:
    #             para=Para(culcType=B12_JACKKNIFE, circuit=circuit, t=t, depth=10, Nq=Nq, Nsample=1000)
    #             data=get_data(para)
    #             if data is None:
    #                 continue
    #             ps_mean=np.mean(data)
    #             vps=np.sum([math.pow(psi-ps_mean, 2) for psi in data]) / (len(data)-1)
    #             conf = 1.960 * math.sqrt(vps/len(data))
    #             csv[f"{Nq}qubit"]["ps"] = ps_mean
    #             csv[f"{Nq}qubit"]["vps"] = vps
    #             csv[f"{Nq}qubit"]["conf"] = conf
    #             csv[f"{Nq}qubit"]["skew"] = data.skew()
    #             csv[f"{Nq}qubit"]["kurt"] = data.kurt()
    #             csv[f"{Nq}qubit"]["result"] = f"{round(ps_mean, 4)}{'±'}{round(conf, 4)}"
    #             # csv[f"{Nq}qubit"]["result"] = (ps_mean, conf)
    #         # print(csv)
    #         csv.to_csv(f'{ROOT_PATH}/result/total/jackknife_{circuit}_t{t}.csv')

    # t_list=[1,2,3,4]
    # for t in t_list:
    #     para=Para(culcType=B12_JACKKNIFE, circuit="RDC", depth=10, t=t, Nq=7, Nsample=1000)
    #     title=f'jackknife_{para.circuit}_Nq{para.Nq}_t{para.t}_sample{para.Nsample}'
    #     data=get_data(para)
    #     ps_mean=np.mean(data)
    #     vps=np.sum([math.pow(psi-ps_mean, 2) for psi in data]) / (len(data)-1)
    #     conf = 1.960 * math.sqrt(vps/len(data))
    #     print(ps_mean, vps, conf, data.skew(), data.kurt())
    #     plt.title(title)
    #     plt.hist(data, bins=30, density=True)
    #     # plt.hist(data, bins=30, density=True, range=[0, math.factorial(t)*2])
    #     plt.axvline(x=[math.factorial(t)], ymin=0, ymax=1, c='y')
    #     plt.axvline(x=[ps_mean], ymin=0, ymax=1, c='r')
    #     plt.axvline(x=[ps_mean-conf], ymin=0, ymax=1, c='r')
    #     plt.axvline(x=[ps_mean+conf], ymin=0, ymax=1, c='r')
    #     plt.savefig(f"{ROOT_PATH}/result/figure/{title}.png")
    #     plt.clf()

def main_bootstrap(alpha=0.95):
    Nq_list=[2,3,4,5,6,7]
    depth_list=[5,6,7,8,9,10,11,12,13,14]
    t_list=[1,2,3,4,5,6]
    # Nsample=200
    rng=np.random.default_rng()
    Ntimes=1000
    Nsample=100
    circuit_list={'LRC', 'RDC'}
    for circuit in circuit_list:
        for t in t_list:
            csv=DataFrame(columns=Nq_list, index=depth_list, dtype=object)
            for Nq in Nq_list:
                for depth in depth_list:
                    para=Para(culcType=B12, circuit=circuit, Ntimes=Ntimes, t=t, depth=depth, Nq=Nq, Nsample=Nsample)
                    data=get_data(para)
                    if data is None:
                        continue
                    ci_mean=bootstrap((data,), np.mean, confidence_level=alpha, n_resamples=10000, random_state=rng, method="BCa")
                    mean=round(ci_mean[1], 3)
                    lci=round(ci_mean[0]-ci_mean[1], 3)
                    rci=round(ci_mean[2]-ci_mean[1], 3)
                    # print(Nq, t, mean, lci, rci)
                    csv[Nq][depth] = f"{mean} +{rci}, {lci}"
            print(csv)
            csv.to_csv(f'{ROOT_PATH}/result/total/bootstrap{alpha}_{circuit}_t{t}_Ntimes{Ntimes}_Nsample{Nsample}.csv', sep="\t")
    
    Nq_list=[2,3,4,5,6,7]
    t_list=[1,2,3,4]
    depth=10
    circuit_list={'RC', 'LRC', 'RDC'}
    rng=np.random.default_rng()
    csv=DataFrame(columns=Nq_list, index=t_list, dtype=object)
    for circuit in circuit_list:
        for Nq in Nq_list:
            for t in t_list:
                para=Para(culcType=B12, circuit=circuit, depth=depth, Ntimes=Ntimes, t=t, Nq=Nq, Nsample=Nsample)
                data=get_data(para)
                if data is None:
                    continue
                ci_mean=bootstrap((data,), np.mean, confidence_level=alpha, n_resamples=10000, random_state=rng, method="BCa")
                mean=round(ci_mean[1], 3)
                lci=round(ci_mean[0]-ci_mean[1], 3)
                rci=round(ci_mean[2]-ci_mean[1], 3)
                print(Nq, t, mean, lci, rci)
                csv[Nq][t] = f"{mean} +{rci}, {lci}"
        csv.to_csv(f'{ROOT_PATH}/result/total/bootstrap{alpha}_{circuit}_Ntimes{Ntimes}_Nsample{Nsample}.csv', sep="\t")

def main_chebyshevs(alpha=0.95):
    Nq_list=[2,3,4,5,6,7]
    # Nq_list=[3,4,5,6,7]
    t_list=[1,2,3,4]
    depth_list=[3,4,5,6,7,8,9,10]
    Ntimes=40
    Nsample=25
    circuit_list={'LRC', 'RDC'}
    for circuit in circuit_list:
        for t in t_list:
            csv=DataFrame(columns=Nq_list, index=depth_list, dtype=object)
            for Nq in Nq_list:
                for depth in depth_list:
                    para=Para(culcType=B12, circuit=circuit, Ntimes=Ntimes, t=t, depth=depth, Nq=Nq, Nsample=Nsample)
                    data=get_data(para)
                    if data is None:
                        continue
                    mean, H=chebyshevs_inequality_confidence_interval(data, k=math.sqrt(1./(1-alpha)))
                    csv[Nq][depth] = f"{round(mean, 3)}{'±'}{round(H, 3)}"
                    # csv[Nq][depth] = (int(mean*1000)/1000, int(H*1000)/1000)
                    # print(Nq, depth, mean, H)
            print(csv)
            csv.to_csv(f'{ROOT_PATH}/result/total/chebyshevs{alpha}_{circuit}_t{t}_Ntimes{Ntimes}_Nsample{Nsample}.csv')

    Nq_list=[2,3,4,5,6,7]
    t_list=[1,2,3,4]
    depth=10
    circuit_list={'RC', 'LRC', 'RDC'}
    csv=DataFrame(columns=Nq_list, index=t_list, dtype=object)
    for circuit in circuit_list:
        for Nq in Nq_list:
            for t in t_list:
                para=Para(culcType=B12, circuit=circuit, depth=depth, Ntimes=Ntimes, t=t, Nq=Nq, Nsample=Nsample)
                data=get_data(para)
                if data is None:
                    continue
                mean, H=chebyshevs_inequality_confidence_interval(data, k=math.sqrt(1./(1-alpha)))
                # mean, H=chebyshevs_inequality_confidence_interval(np.sort(data)[100:-100], k=3)
                print(Nq, t, mean, H)
                csv[Nq][t] = f"{round(mean, 3)}{'±'}{round(H, 3)}"
                # csv[Nq][t] = (int(mean*1000)/1000, int(H*1000)/1000)
        csv.to_csv(f'{ROOT_PATH}/result/total/chebyshevs{alpha}_{circuit}_Ntimes{Ntimes}_Nsample{Nsample}.csv')

if __name__ == "__main__":
    alpha=[0.89, 0.95]
    for a in alpha:
        # main_chebyshevs(alpha=a)
        main_bootstrap(alpha=a)
    # main_jackknife()
    # test()
