# #!/bin/bash


function check_python_module_exists() {
	[ $(python -c "import $1; print($1.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ]
}

function isntall_python_module() {
	if check_python_module_exists $2;
	then
		echo -e "$(tput setaf 3)already installed $(tput sgr 0)"
	else
		OUTPUT=$(sudo -H $3 install $1 2>&1)
		if check_python_module_exists $2;
		then
			echo -e "$(tput setaf 2)success $(tput sgr 0)"
		else
			echo -e "$(tput setaf 1)fail $(tput sgr 0)"
			echo -e ${OUTPUT}
		fi	
	fi
}


cd ~
mkdir git &>/dev/null
cd git
sudo apt-get update &>/dev/null

# install opencv
echo -e "$(tput setaf 6)INSTALLING OpenCV AND REQUIREMENTS...$(tput sgr 0)"

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
   			echo -e "$(tput setaf 1)fail$(tput sgr 0)"
   			echo -e $OUTPUT
   		else 
			echo -e "$(tput setaf 2)success$(tput sgr 0)"
   		fi
   	else 
   		echo -e "$(tput setaf 3)already installed$(tput sgr 0)"
	fi
done

PYTHON_MODULES=('matplotlib' 'numpy')

for MODULE in ${PYTHON_MODULES[@]}
do
	echo -n "Installing $MODULE..."
	isntall_python_module $MODULE $MODULE "pip2"	
done

continue_opencv_install="true"
if [ $(python -c "import cv2; print(cv2.__version__); print(help(cv2.dnn));" 2>/dev/null |& grep -c "Error:") -eq 0 ];
then
	echo -e "$(tput setaf 3)OpenCV already isntalled$(tput sgr 0)"
	continue_opencv_install="false"
fi

if [[ "$continue_opencv_install" == "true" ]]
then
	echo -n "Cloning opencv repo..."
	OUTPUT=$(git clone https://github.com/opencv/opencv.git 2>&1)
	if [ $(cd opencv |& grep -c "cd:") -eq 0 ]
	then
		echo -e "$(tput setaf 2)ok$(tput sgr 0)"
	else
		continue_opencv_install="false"
		echo -e "$(tput setaf 1)fail$(tput sgr 0)"
		echo -e ${OUTPUT}
	fi

	echo -n "Cloning opencv_contrib repo..."
	OUTPUT=$(git clone https://github.com/opencv/opencv_contrib.git	2>&1)
	if [ $(cd opencv_contrib |& grep -c "cd:") -eq 0 ]
	then
		echo -e "$(tput setaf 2)ok$(tput sgr 0)"
	else
		continue_opencv_install="false"
		echo -e "$(tput setaf 1)fail$(tput sgr 0)"
		echo -e ${OUTPUT}
	fi
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
   		echo -e "$(tput setaf 2)success$(tput sgr 0)"
   		echo -e "$(tput setaf 2)OpenCV installed successfully$(tput sgr 0)"
	else
		echo -e "$(tput setaf 1)fail$(tput sgr 0)"
		echo -e ${OUTPUT}
	fi
fi

cd ../..

# install dlib
echo -n "$(tput setaf 6)INSTALLING DLIB ...$(tput sgr 0)"
isntall_python_module "dlib" "dlib" "pip"

# install openface
echo -n "$(tput setaf 6)INSTALLING OPENFACE...$(tput sgr 0)"
need_install_openface="true"
if [ $(python -c "import openface;" 2>/dev/null |& grep -c 'Error') -eq 0 ];
then 
	echo -e "$(tput setaf 3)already isntalled$(tput sgr 0)"
	need_install_openface="false"
fi
if [[ "$need_install_openface" == "true" ]]
	then
	OUTPUT=$(git clone https://github.com/cmusatyalab/openface.git 2>&1)
	if [ $(cd openface |& grep -c "cd:") -eq 0 ]
	then 
		cd openface
		OUTPUT=$(pip install -r requirements.txt && sudo python setup.py install 2>&1)	
		if [ $(python -c "import openface;" 2>/dev/null |& grep -c 'Error') -eq 0 ];
		then
	   		echo -e "$(tput setaf 2)success$(tput sgr 0)"
		else
			echo -e "$(tput setaf 1)fail$(tput sgr 0)"
			echo -e ${OUTPUT}
		fi
		cd ..
	else
		echo -e "$(tput setaf 1)fail$(tput sgr 0)"
		echo -e ${OUTPUT}
	fi
fi

# install face recognition model
echo -n "$(tput setaf 6)INSTALLING FACE_RECOGNITION_MODELS...$(tput sgr 0)"
isntall_python_module "git+https://github.com/ageitgey/face_recognition_models" "face_recognition_models" "pip"

# install requirements for face repo
echo -n "$(tput setaf 6)INSTALLING FLASK...$(tput sgr 0)"
isntall_python_module "flask" "flask" "pip"


echo -n "$(tput setaf 6)INSTALLING TENSORFLOW...$(tput sgr 0)"
isntall_python_module "tensorflow" "tensorflow" "pip"


echo -n "$(tput setaf 6)INSTALLING KERAS...$(tput sgr 0)"
isntall_python_module "--no-cache-dir keras" "keras" "pip"
