# longview-deep-learning
# Some simple reinforcement learning examples for longview students.
#
# Preparation instructions for MacOS:  (best of luck for windows users... (it shouldn't be too hard))
# In general you will need python3.6, tensorflow, gym.  You can install those any way
# you want or you can follow the instructions below for installing python via homebrew.
# Instructions are below.  I don't have a blank machine to test this on but it 
# will probably work.  :)
# you can simply type "bash README.md" as this README is also a bash script.


/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install python3
echo
echo "this should show python 3.6.4 or later"
python --version
echo
echo
pip3 install virtualenv
virt_dir=~/virtualenvs/tf_dev
virtualenv -p /usr/bin/python3.6 $virt_dir
pip install -r requirements.txt

echo "You will need to 'activate' your script by typing (you can use cut/paste to paste the line):"
echo "source activate $virt_dir/bin/activate"


# To run the pong game:
# . $virt_dir/bin/activate
# cd pong
# python pong.py
# <or>
# python saved_pong.py --checkpoint_dir saved_model

# to run the apples game:
# . $virt_dir/bin/activate
# cd apples
# python interactive_apples.py
# or
# python batch_apples.py
# take a look at interactive_apples.py: on_key_press() to see keyboard commands
