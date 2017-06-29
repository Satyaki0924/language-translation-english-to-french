# Language Translator (English - French)
This project uses sequence to sequence model of recurrent neural network to translate any piece of English text to French. I have used recurrent nets because while training on huge data, recurrent nets actually predict the outcome a lot better than any normal machine learning models. In this specific model, the data first passes through an encoder, comes out as an understanding and passes to a decoder. The decoder generates the output.

##### *** This project will throw errors if trained on CPU instead of GPU ***

![Terminal screen_error](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/lt_error.png?raw=true "Terminalerror")

### This project is configured for Linux and uses python3
To run this project, open up your bash terminal and write

```
chmod -R 777 setup.sh
```
This will set up the project enviornment for you. This must be run with administrator rights.

```
./setup.sh
```

#### * Virtual enviornment will be setup and activated for you

##### ** (If not activated, use the following command) **

```
source venv/bin/activate
```
Install the required packages using the following command.
```
pip install -r requirements.txt
```

## Train the project

```
python run_me.py
```

![Terminal screen_1](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/lt1.png?raw=true "Terminal1")

##### ** Choose 1 to train **

![Terminal screen_2](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/lt2.png?raw=true "Terminal2")

![Terminal screen_3](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/lt3.png?raw=true "Terminal4")

## Test the project
Run the python file, following the instructions

```
python run_me.py
```

The outcome should look something like this:

![Terminal screen_4](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/lt4.png?raw=true "Terminal4")

## Plotting the graphs

#### ** If the training goes well, the graphs should look something like this **

![Plot](https://github.com/Satyaki0924/language-translation-english-to-french/blob/master/res/plot.png?raw=true "Plot")

### Author: Satyaki Sanyal
#### *** This project is strictly for educational purposes only. ***
