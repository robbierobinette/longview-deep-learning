# longview-deep-learning
# Some simple reinforcement learning examples for longview students.
#
# Preparation instructions for MacOS:  (best of luck for windows users... (it shouldn't be too hard))
# In general you will need python3.6, tensorflow, gym.  You can install those any way
# you want or you can follow the instructions below for installing python via homebrew.
# I don't have a blank machine to test this on but it will probably work.  :)
# you can simply type "bash README.md" as this README is also a bash script.
#
# This may take a while as it will have to download and install Xcode if it is not already
# installed on your machine. 


/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install --upgrade python3
echo
echo "this should show python 3.6.4 or later"
python --version
echo
echo
brew install cmake
pip3 install virtualenv
virt_dir=~/virtualenvs/tf_dev
virtualenv -p /usr/bin/python3.6 $virt_dir
pip install -r requirements.txt

echo "You will need to 'activate' your script by typing (you can use cut/paste to paste the line):"
echo "source activate $virt_dir/bin/activate"


# To run the pong game.  This will start with an untrained network.  It will take a while (several hours at least)
# to train.  You can uncomment the 'p.render()' line to watch it though that is *slow*.  if you stop it and 
# restart the game it will pick up from where it left off (mostly).  On a restart it has to repopulate the memory before
# it can resume learning.
#
# . $virt_dir/bin/activate
# cd pong
# python pong.py

# This will show you what a trained model looks like when playing pong.  It's pretty good but is clearly exploiting
# a weakness in the opponent.
#
# python saved_pong.py --checkpoint_dir saved_model
#
# Another use of the saved_pong.py is to check in on the progress of
# the learning.  You can specify the checkpoint directory and it will
# read the latest checkpoint of the running pong game and use that
# to play.  This does not interfere with the learning job.
#
# python saved_pong.py --checkpoint_dir ../checkpoints/pong-simple

# to run the apples game in interactive mode:
# . $virt_dir/bin/activate
# cd apples
# python interactive_apples.py
# take a look at interactive_apples.py: on_key_press() to see keyboard commands
#
# this runs the apples game without doing any rendering.  One 'to do'
# item for me is to refactor the apples_game into something which has
# the same interface as the openai stuff.  It's a little tricky, I'll
# have to deal with the rendering stuff in some special way as just
# importing the graphics library causes it to fail if it cannot find
# display.  That's why I have batch_apples.py, it never imports the
# graphics library (arcade).  

# python batch_apples.py

