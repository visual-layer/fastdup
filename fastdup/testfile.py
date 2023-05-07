SELECTION_STRATEGY_FIRST=0
SELECTION_STRATEGY_RANDOM=1
SELECTION_STRATEGY_UNIFORM_METRIC=2
import random

def sample_from_components(row, metric, kwargs, howmany):
    selection_strategy = kwargs['selection_strategy']
    if selection_strategy == SELECTION_STRATEGY_FIRST:
        return row['files'][:howmany]
    elif selection_strategy == SELECTION_STRATEGY_RANDOM:
        return random.sample(row['files'], howmany)
    elif selection_strategy == SELECTION_STRATEGY_UNIFORM_METRIC:
        assert metric in row, "When using selection_strategy=2 (SELECTION_STRATEGY_UNIFORM_METRIC) need to call with metric=metric."
        assert len(row[metric]) == len(row['files'])
        #Combine the lists into a list of tuples
        combined = zip(row['files'], row[metric])

        # Sort the list of tuples by the float value
        sorted_combined = sorted(combined, key=lambda x: x[1])
        print(sorted_combined)

        sindices = range(0, len(sorted_combined), int(len(sorted_combined)/howmany))
        print(sindices)

        # Extract the filenames from the selected subset
        filenames = [sorted_combined[t][0] for t in sindices]
        return filenames

if __name__ == "__main__":
    #print(get_bounding_box_func_helper("../t1/atrain_crops.csv"))
    file = ["a","b","c","d", "e","f","g","h"]
    floats = [1,2,3,4,1,2,3,1]
    row = {}
    row['blur'] = floats
    row['files'] = file
    files = sample_from_components(row, 'blur', {'selection_strategy':2}, 2)
    print(files)