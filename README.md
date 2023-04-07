# Semi_supervised

This file can create a semi-supervised data set in one of two ways. 

1) The standard way. We iterate through the code at hand and we predict on unlabled data, some label. If this label is within some confidence interval, we update the
label and append it to the data. We continue iteration til we include no new data. 

2) The new way. We iterate through the code at hand and predict on unlabled data. Then as we add to the unlabled data, our features include some statistics
That describe the data up to some time. We now update those statistics at hand. This is more sophisticated approach found in lines 288- 289



#runner.py

This file can be used to iterate the semi-supervised model n times. Often we will need to create
