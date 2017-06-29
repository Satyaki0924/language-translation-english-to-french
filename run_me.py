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
