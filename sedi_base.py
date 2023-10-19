from scipy.linalg import toeplitz,solve_toeplitz
from scipy.signal import correlate,convolve
from scipy.fft import fft,ifft
import numpy as np

from obspy import Trace
import obspy


def rebel_rf(trace: np.ndarray, *args, **kwargs):
    """
    this method is developed from method proposed by 
    Yu,Y.,J.Song,K.H.Liu,and S. S. Gao (2015), Determining crustal structure beneath seismic stations overlying a low-velocity sedimentary layer using receiver functions, J. Geophys. Res. Solid Earth, 120, 3208–3218, doi:10.1002/2014JB011610.
    
    may be simple and rubost
    """
    Fs = 1 / kwargs.get("delta", 0.1)
    size = trace.shape[0]
    omega = 2 * np.pi * np.arange(size) * Fs / size

    method = kwargs.get("method", "direct")
    cor = correlate(trace, trace,
                    method=method, mode="full")[size - 1:]
    cor = cor / cor[0]

    dt = _pick_measure(cor)
    r0 = -cor[dt]
    #print(r0, dt)
    dt /= Fs

    filter = 1 + r0 * np.exp(-1j * omega * dt)
    return ifft(np.multiply(fft(trace), filter)).real


def predict_RF(trace, *args, **kwargs):
    """
    this method is developed from 朱洪翔 et.al 2018 地球物理学报
    here step represent m in equation 5,
    m+a=接收函数总长
    a值略大于多次波周期，最好非常接近。
    """
    size = trace.shape[0]
    m = kwargs.get("m", 30)
    b = kwargs.get("b", 0.001)
    trace[0] *= (1 + b)
    method = kwargs.get("method", "direct")
    cor = correlate(trace, trace,
                    method=method, mode="full")[size - 1:]

    a = _pick_measure(cor)
    c_mat = solve_toeplitz((cor[0:m], cor[0:m]), cor[a: a + m])

    d = np.zeros(a + m)
    d[0] = 1
    d[a:] = (-1 * c_mat)

    return convolve(d, trace.data, mode="full", method=method)

def _pick_measure(conv):
    """
    get first trough and first pick in conv
    Parameters
    ----------
    conv : correlated RF
    """
    sig = np.sign(np.diff(conv))
    sig = np.diff(sig)

    return np.where(sig > 0)[0][0] + 1

if __name__ == "__main__":
    from os.path import dirname,join
    file="/home/jous/Desktop/F/project/public/X2.15683/remove_sedi.lst"
    dir = dirname(file)
    s1_dir=join(dir,"rebel")
    s2_dir=join(dir,"predict")
    link=open(file,'r')
    file = link.readlines()
    link.close()


    for _i in file:
        _i=_i[:-1]
        trace = obspy.read(join(dir,_i))[0]
        rebel = trace.copy()
        predict = trace.copy()

        rebel.data = rebel_rf(rebel.data)
        predict.data = predict_RF(predict.data)

        rebel.write(join(s1_dir,_i),format="SAC")
        predict.write(join(s2_dir,_i),format="SAC")



