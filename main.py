import train
import test


def main():
    dataTrain, labelsTrain = train.train_imgToArray()
    cowsTrain, labelsTrain = train.train_convertToNumpy(dataTrain, labelsTrain)
    x_train, y_train = train.train_createData(cowsTrain, labelsTrain)
    model = train.kerasModel()
    model = train.training(model, x_train, y_train)

    dataTest, labelsTest = test.test_imgToArray()
    cowsTest, labelsTest = test.test_convertToNumpy(dataTest, labelsTest)
    x_test, y_test = test.test_createData(cowsTest, labelsTest)
    test.testing(model, x_test, y_test)


if __name__ == "__main__":
    main()