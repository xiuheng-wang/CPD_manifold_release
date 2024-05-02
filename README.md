# Non-parametric Online Change Point Detection On Riemannian manifolds

Steps:

1. Run main_spd.py to plot Figure 1, Figure 5 (left) and Figure 6 (left);

2. Run main_grassmann.py to plot Figure 2, Figure 5 (right) and Figure 6 (right);

3. Download the <a href="https://figshare.com/articles/dataset/TIMIT_zip/5802597"> TIMIT </a> and <a href="https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/"> QUT-NOISE </a> database, convert WAV files <a href="https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python"> from stereo to mono </a>, and run main_vad.py to plot Figure 3;

4. Download the <a href="https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/HDM05_SPDData.zip"> HDM05 </a> SPD data, and run main_sar.py to plot Figure 4 and Figure 7;

5. Run main_distribution.py to plot Figure 8;

6. Run main_adapt_thres.py to plot Figure 9.

For any questions, feel free to email us at xiuheng.wang@oca.eu or raborsoi@gmail.com.

If these codes are helpful for you, please cite our paper as follows:

    @inproceedings{wang2024nonparametric,
      title={Non-parametric Online Change Point Detection on Riemannian Manifolds},
      author={Wang, Xiuheng and Borsoi, Ricardo Augusto and Richard, C{\'e}dric},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2024},
      organization={PMLR}
    }

Note that the copyright of the manopt toolbox is reserved by https://pymanopt.org/.
