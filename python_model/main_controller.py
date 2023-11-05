import os
from folders import folder_arff, folder_code
import shutil

######################################
# grayscale images or rgb
imageChannels=1

# image size 
params = {'dim': (128,128,imageChannels),
          'batch_size': 4,
          'shuffle': True,
          'resolution' : 1}

datasetpath =os.path.abspath("dataset9")

######################################
# OVERVIEW
# 1 data --> angular velocities
# 2 angular velocities --> scarce distributions 
# 3 scarce distributions --> contours
# 4 train envoder contours  

################################################
################################################         
# 1 load data from dataset with a uniform format 
# data is loaded and trnaformed into:
# "GazeFoVDegreesX", "GazeFoVDegreesY", 
# "headVelX", "headVelY",
# "headVelAng","headVelMagnitude",
# "headAccX","headAccY","headAccAng",
# "headAccMagnitude"
# the output a single binary file named HeadVelocityAndGaze.pkl

python_script= "generateParameters.py"
filename = "360_VR_gaze.pkl"
arff_headVelocityandgaze_file =  os.path.join(datasetpath,filename)
command = "python {0} --path {1} --type arff --output {2}".format(python_script,folder_arff,arff_headVelocityandgaze_file)
#os.system(command)


#todo add pure data and s-gaze
#1 bis, copy confirgurations
src = os.path.join(folder_code, 'FixedArff.py')
dest = os.path.join(datasetpath, 'FixedArff.py')
shutil.copy(src,dest)
src = os.path.join(folder_code, 'heatmap.py')
dest = os.path.join(datasetpath, 'heatmap.py')
shutil.copy(src,dest)

##############################################
##############################################
# 2 create real data distributions 
# this create the distribution images in outputdataset\\distribution 
# this also fit a 2D gaussian distribution in outputdataset\\gaussian 

python_script= "FixedArff.py"
realdataset = os.path.join(datasetpath,"datasetreal")
command = "python {0} --file {1} --outputpath {2} --resolution {3}".format(python_script, arff_headVelocityandgaze_file,realdataset,params['dim'][0])
#os.system(command)

##############################################
##############################################
# 3 Train autoencoder and predict distributions 

# 1st create distributions 
python_script= "deeplearning\\distributions.py"
command = "python {0} --path {1} --resolution {2}".format(python_script,datasetpath,params['dim'][0])
#os.system(command)

# 2nd train autoencoder #TODO add conf parameters
python_script= "deeplearning\\autoencoder_training.py"
command = "python {0} --rootpath {1} --resolution {2} --imageChannels {3}".format(python_script,datasetpath,params['dim'][0],imageChannels)
#os.system(command)

# 3rd predict distributions from the real data scarce distributions 
python_script= "deeplearning\\autoencoder_prediction.py"
command = "python {0} --rootpath {1} --resolution {2} --imageChannels {3}".format(python_script,datasetpath,params['dim'][0],imageChannels)
#os.system(command)

#3rd bis # TODO replace with afm autogeneration 
src = os.path.join(folder_code, 'afm')
dest = os.path.join(realdataset, 'afm')
try:
    shutil.copytree(src, dest) 
except:
    print('afm already present')

# 4th evaluate distributions comapring it to gaussian 
python_script= "compareDistributions.py"
command = "python {0} --path {1}".format(python_script,realdataset)
#os.system(command)


##############################################
##############################################
# 4 extract contours from continus distributions and evaluate extracted contours

python_script= "contours.py"
command = "python {0} --path {1} --GazeHeadFile {2}".format(python_script,datasetpath,arff_headVelocityandgaze_file)
#os.system(command)


# python_script= "generate_evaluation_distributions.py"
# realdataset = os.path.join(datasetpath,"datasetreal")
# command = "python {0} --path {1} --GazeHeadFile {2}".format(python_script,datasetpath,arff_headVelocityandgaze_file)
#os.system(command)

# python_script= "evaluate_contours.py"
# command = "python {0} --path {1} --GazeHeadFile {2} --outEvaluate evaluate.csv".format(python_script,datasetpath,arff_headVelocityandgaze_file)
#os.system(command)


##############################################
##############################################
# 5 training contours encoder

python_script= "deeplearning/countour_encoder.py"
command = "python {0} --path {1}".format(python_script,datasetpath)
os.system(command)


##############################################
##############################################
# 6 evaluate contours encoder

python_script= "deeplearning/countour_encoder_prediction.py"
command = "python {0} --path {1} --GazeHeadFile {2}".format(python_script,datasetpath,filename)
#os.system(command)


python_script= "deeplearning/keras2tflite.py"
command = "python {0} --modelpath {1} ".format(python_script,datasetpath)
#os.system(command)
