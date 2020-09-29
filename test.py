import tools


# -------------------------- Prepare Test Set --------------------------
# Making images into array
def test_imgToArray():
    rootdirTest = '.\dataset\\test'
    dataTest, labelsTest = tools.imgToArray(rootdirTest)
    return dataTest,labelsTest


# Convert dataTest and labelsTest to numpy arrays
def test_convertToNumpy(dataTest, labelsTest):
    cowsTest, labelsTest = tools.convertToNumpy(dataTest, labelsTest)
    return cowsTest, labelsTest


# Create data - X and Y
def test_createData(cowsTest, labelsTest):
    x_test, y_test = tools.createData(cowsTest, labelsTest)
    return x_test, y_test


# -------------------------- Test the model --------------------------
def testing(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n', 'Test accuracy:', score[1])
