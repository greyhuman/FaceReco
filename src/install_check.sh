#!/bin/bash


MODULES=('build-essential' 'cmake' 'git' 'libgtk2.0-dev' 'pkg-config'
			'libavcodec-dev' 'libavformat-dev' 'libswscale-dev'  'python-dev'
			 'python-numpy' 'libtbb2' 'libtbb-dev' 'libjpeg-dev' 'libpng-dev' 
			 'libtiff-dev' 'libdc1394-22-dev' 'python-pip' 'python-tk')

for MODULE in ${MODULES[@]}
do
	echo -n "check" $MODULE  "installed..."
	if [ $(dpkg-query -W -f='${Status}' $MODULE 2>/dev/null | grep -c "ok installed") -eq 0 ];
	then
   		echo -e "$(tput setaf 1)no$(tput sgr 0)"
	else
		echo -e "$(tput setaf 2)yes$(tput sgr 0)"
	fi
done


echo -e
echo -e


PYTHON_MODULES=('matplotlib' 'cv2' 'dlib' 'openface' 'face_recognition_models' 'flask' 'numpy' 'keras' 'tensorflow')


for MODULE in ${PYTHON_MODULES[@]}
do
	echo -n "check" $MODULE  "installed..."
	COMMAND=""
	if [ $MODULE == 'openface' ];
	then
		COMMAND="import $MODULE;"
	elif [ $MODULE == 'cv2' ]; 
	then
		COMMAND="import $MODULE; print($MODULE.__version__); print(help($MODULE.dnn));"
	else
		COMMAND="import $MODULE; print($MODULE.__version__)"
	fi
	if [ $(python -c "$COMMAND" 2>/dev/null |& grep -c "Error:") -eq 0 ];
	then
   		echo -e "$(tput setaf 2)yes$(tput sgr 0)"
	else
		echo -e "$(tput setaf 1)no$(tput sgr 0)"
	fi
done

