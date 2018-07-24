import random
#generates fake data from a 1D list
#returns a new list with fake data
def generate_fake_list(values):
    max_val = max(values)
    min_val = min(values)
    #keeps track of the maximum and minimum values from the list to ensure that the fake data does not escape these bounds
    new_values = list()
    variation = 1
    #[variation] represents the scale factor by which values are multiplied
    #this value changes dynamically
    for i in range(len(values)):
        if(random.uniform(0,1)>0.9):
            random.seed()
            #resets the random library 10% of the time to ensure continued randomness
        new_values.append(values[i] * variation)
        if(new_values[i]>max_val):
            variation -= 0.05
        elif(new_values[i]<min_val):
            variation += 0.05
            #pushes the variation up or down depending on whether the new value has been pushed above or below the minimum
        elif(variation > 1.2):
            variation -= 0.05
        elif(variation < 0.8):
            variation += 0.05
            #pushes the variation up or down depending on whether the variation has been pushed too high or too low
        else:
            variation += random.uniform(-0.015,0.015)
            #changes the variation index by a random value
    return new_values

#generates fake data for a list of paired data values (e.g. a list of tuples containing temperature, light, and humidity)
#this is done by creating a separate list for each individual variable, using generate_fake_list to generate fake data for that variable, and then returning the data to its original format
def generate_fake(data_list):
    #data_list contains data as a list of x-value tuples
    sample_tuple = data_list[0]
    x = len(sample_tuple)
    array_2d = list()
    #array_2d will be a list of x lists
    for i in range(x):
        array = list()
        for item in data_list:
            array.append(item[i])
        array = generate_fake_list(array)
        #generates fake data within the array
        array_2d.append(array)
    new_array = list()
    sample_list = array_2d[0]
    list_len = len(sample_list)
    #data converted back to list of x-value pairs
    for i in range(list_len):
        paired_values = list()
        for arr in array_2d:
            paired_values.append(arr[i])
        new_array.append(paired_values)
    return new_array
