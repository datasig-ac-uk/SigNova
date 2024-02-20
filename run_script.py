from SigNova import *

start_time = time.time()

flagger = SigNova()

flagger.create_corpus()

flagger.fit(show_progress=True)

flagger.create_inliers() #For Training/calibration

flagger.calibrate(show_progress=True) #For Training/calibration

name = 'RealData'
alpha = 'alpha0005'

flagger.get_inliers_scores(name='Scores_{0}_{1}'.format(name,alpha))


flagger.create_test()


flagger.flag(processes=14, output_name='npy/Pysegment_{0}_{1}'.format(name,alpha)) #, all_inliers=True)
flagger.plot_result(flagger.arr, outname='plots/Pysegment_{0}_{1}'.format(name,alpha), telescope='MWA')


##### To test over more than one dataset:
###
#base_dir  = '/scratch/radioastro/data/'
##
#files = [
# 'RealData_1061319104.pkl',\
# 'RealData_1061319224.pkl',\
# 'RealData_1061319352.pkl']
#
#for i, file in enumerate(files):
#    flagger.create_test(data_path=base_dir+file)
#    flagger.flag(processes=14, output_name='npy/NewPysegments_Alpha0005'+str(i)) #, all_inliers=True)
#    flagger.plot_result(flagger.arr, outname='plots/NewPysegments_Alpha0005'+str(i), telescope='MWA')
