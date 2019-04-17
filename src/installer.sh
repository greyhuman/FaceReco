# #!/bin/bash

cd ~
mkdir git &>/dev/null
cd git
sudo apt-get update 2>/dev/null

# install opencv
echo -e "Installing opencv..."
MODULES=('build-essential' 'cmake' 'git' 'libgtk2.0-dev' 'pkg-config'
			'libavcodec-dev' 'libavformat-dev' 'libswscale-dev'  'python-dev'
			 'python-numpy' 'libtbb2' 'libtbb-dev' 'libjpeg-dev' 'libpng-dev' 
			 'libtiff-dev' 'libdc1394-22-dev' 'python-pip' 'python-tk')


for MODULE in ${MODULES[@]}
do
	echo -n "Installing $MODULE..."
	if [ $(dpkg-query -W -f='${Status}' $MODULE 2>/dev/null | grep -c "ok installed") -eq 0 ];
	then
   		OUTPUT=$(sudo apt-get -y install $MODULE  2>&1)
   		if [ $(dpkg-query -W -f='${Status}' $MODULE 2>/dev/null | grep -c "ok installed") -eq 0 ];
   		then
   			echo -e "fail"
   			echo -e $OUTPUT
   		else 
			echo -e "success"
   		fi
   	else 
   		echo -e "already installed"
	fi
done

echo -n "Installing matplotlib..."
OUTPUT=$(sudo pip2 install matplotlib 2>&1)
if [ $(python -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "success"
else
	echo -e 
	echo -e ${OUTPUT}
fi

echo -n "Installing numpy..."
OUTPUT=$(pip2 install numpy 2>&1)
if [ $(python -c "import numpy; print(numpy.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "success"
else
	echo -e 
	echo -e ${OUTPUT}
fi

continue_opencv_install="true"
echo -n "Cloning opencv repo..."
OUTPUT=$(git clone https://github.com/opencv/opencv.git 2>&1)
if [ $(cd opencv |& grep -c "cd:") -eq 0 ]
then
	echo -e "ok"
else
	continue_opencv_install="false"
	echo -e "fail"
	echo -e ${OUTPUT}
fi

echo -n "Cloning opencv_contrib repo..."
OUTPUT=$(git clone https://github.com/opencv/opencv_contrib.git	2>&1)
if [ $(cd opencv_contrib |& grep -c "cd:") -eq 0 ]
then
	echo -e "ok"
else
	continue_opencv_install="false"
	echo -e "fail"
	echo -e ${OUTPUT}
fi

if [[ "$continue_opencv_install" == "true" ]]
then
	cd opencv
	git branch –a &>/dev/null
	git checkout 3.4 &>/dev/null
	mkdir build &>/dev/null
	cd build
	echo -n "Build opencv(it will take a long time, about half an hour)..."
	OUTPUT=$(cmake -D CMAKE_BUILD_TYPE=RELEASE –D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_LIBV4L=ON –D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ .. 2>&1 && make 2>&1 && sudo make install 2>&1)
	if [ $(python -c "import cv2; print(cv2.__version__)"  2>/dev/null |& grep -c 'Error') -eq 0 ];
	then
   		echo -e "success"
	else
		echo -e "fail"
		echo -e ${OUTPUT}
	fi
fi

cd ../..

# install dlib
echo -n "Installing dlib..."
OUTPUT=$(pip install dlib 2>&1)
if [ $(python -c "import dlib; print(dlib.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "success"
else
	echo -e "fail"
	echo -e ${OUTPUT}
fi

# install openface
echo -n "Installing openface..."
OUTPUT=$(git clone https://github.com/cmusatyalab/openface.git 2>&1)
if [ $(cd openface |& grep -c "cd:") -eq 0 ]
then 
	cd openface
	OUTPUT=$(pip install -r requirements.txt && sudo python setup.py install 2>&1)	
	if [ $(python -c "import openface;" 2>/dev/null |& grep -c 'Error') -eq 0 ];
	then
   		echo -e "success"
	else
		echo -e "fail"
		echo -e ${OUTPUT}
	fi
	cd ..
else
	echo -e "fail"
	echo -e ${OUTPUT}
fi

# install face recognition model
echo -n "Installing face_recognition_models..."
OUTPUT=$(pip install git+https://github.com/ageitgey/face_recognition_models 2>&1)
if [ $(python -c "import face_recognition_models; print(face_recognition_models.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "success"
else
	echo -e "fail"
	echo -e ${OUTPUT}
fi

# install requirements for face repo
echo -n "Installing flask..."
OUTPUT=$(pip install flask 2>&1)
if [ $(python -c "import flask; print(flask.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "success"
else
	echo -e "fail"
	echo -e ${OUTPUT}
fi
