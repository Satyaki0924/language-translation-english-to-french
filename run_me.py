"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.
Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""
from src.controller import Controller


def main():
    while True:
        try:
            choice = int(input('>> Enter 1. to Train, 2. to Translate, 3. to plot graph, 4. to Exit! \n'))
            if choice == 1 or choice == 2 or choice == 3:
                Controller.main(choice)
            elif choice == 4:
                break

        except Exception as e:
            print('*** Error: ' + str(e) + '. Try Again ***')
            pass


if __name__ == '__main__':
    main()
