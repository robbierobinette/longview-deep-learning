# longview-deep-learning
#some simple reinforcement learning examples for longview students


# Preparation instructions for macos:
# In general you will need python3.6, tensorflow, gym 
# The most difficult part is installing homebrew.  
# If you have not done this it is fairly easy, you can type the things below or 
# you can simply type "bash README.md" as this README is also a bash script.


/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install python3
echo
echo "this should show python 3.6.4"
python --version
echo
echo
pip3 install virtualenv
virt_dir=~/virtualenvs/tf_dev
virtualenv -p /usr/bin/python3.6 $virt_dir
pip install -r requirements.txt

echo "You will need to 'activate' your script by typing (you can use cut/paste to paste the line):"
echo "source activate $virt_dir/bin/activate"

