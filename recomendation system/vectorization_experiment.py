import itertools

def function_plugger(fun,fun_dict):
    parameter_name = list()
    iterating_values = list()
    for key, value in fun_dict.items():
        parameter_name.append(key)
        iterating_values.append(value)
    enumerated_values = list(itertools.product(*iterating_values))
    possible_functions = list()
    for combo in enumerated_values:
        parameters = dict(zip(parameter_name, combo)) #this may be enumerated
        possible_functions.append(fun(**parameters))
    return (possible_functions, enumerated_values)

def fun(a, b, c):
    return lambda d: d*a*b *c

fun_dict = {'a':[1, 2, 3, 4], 'b':[2, 3, 4], 'c':[6, 9] }
(possible_functions, enumerated_values) = function_plugger(fun,fun_dict)
print(enumerated_values)
print(possible_functions[0](1))