import os
os.environ["MKL_NUM_THREADS"]     = "1" #These can be overwritten if more threads wanted -> specify in the function using threads
os.environ["NUMEXPR_NUM_THREADS"] = "1" #This just sets a default value
os.environ["OMP_NUM_THREADS"]     = "1"

import sys;
import matplotlib.pyplot as plt;
import math;
import numpy as np;

"""
. This program takes output data of various formats and reads the data
. It then stores this data for graphing or other use
"""

def read_singleInterCalc_data(filename):
    """
    . Reads the data from a single interation set calculation
    . This is; 1 particle is acted on by N other particles (all randomly placed here for better average performance result), and the resulting updated particle output
    
    . This program accepts a single file containing lines of data formatted as such;
    <particles>,<threads>,<full_time>
    . Each line is stored in a dictionary
    . Multiple of these lines can be read and stored as list of dictionaries; [{}, {}, ...]
    """
    datafile = open(filename,"r");  #Data located inside same folder as this program
    #NOTE; This lookup MUST be changed when considering regular data vs data froma  mixed system (just OpenMP or MPI vs Both)
    """
    #Non-Mixed data
    parameter_lookup = {
        0 : "particles",
        1 : "threads",
        2 : "full_time",
    };
    """
    #Mixed data
    parameter_lookup = {
        0 : "particles",
        1 : "ranks",
        2 : "full_time",
        3 : "threads"
    };
    data_parameters_set = [];
    for line in datafile:
        data_parameters = {};
        parameter_number = 0;
        start_index = 0;
        for char_index in range(len(line)):
            character = line[char_index: char_index+1];
            match character:
                case ',':
                    value = float(line[start_index:char_index]);
                    value_title = parameter_lookup[parameter_number];
                    data_parameters.update({value_title:value});
                    start_index = char_index+1;
                    parameter_number+=1;
        if(len(data_parameters) > 0):   #If has had no troubles adding values to this (prevents errors from empty lines)
            data_parameters_set.append(data_parameters);
    return data_parameters_set;

def get_colour_gradient(N):
    """
    . Returns a set of N colours gradiented between 0 and a max value given
    """
    data_colour = [];
    for i in range(N):
        cColour = (math.sin( math.pi*(i/N) ), i/N, 1.0-math.sin( math.pi*(i/N) ));
        data_colour.append(cColour);
    return data_colour;

def operate_on_data_set(data_set, operation):
    """
    . Performs an operation on a data_set
    . 'operation' is a dictionary such that;
        operation["name"] = Name of the operation, present for all operations
        operation[...] = Other parameters needed by operation to specify it fully

    . Returns data-set, and also changes data_set by reference (some operations however not affected by reference change, such as in average_ordered)
    """
    if(operation["name"]=="average_ordered"):
        """
        . Finds the average of the next N parameters in the set, combines into a single parameter
        """
        variable_target = operation["variable_target"];
        average_length  = operation["average_length"];
        new_data_set = [];
        counter = average_length;
        for i in range(len(data_set)):   #Look through data set, perform operation where required
            if(counter%average_length==0):
                #When the length is reached, package values together, move onto next term
                #print("     Cond hit, i="+str(i));
                new_data_set.append(data_set[i]);
                new_data_set[len(new_data_set)-1][variable_target] = data_set[i][variable_target] / average_length;   #Set average start
            else:
                new_data_set[len(new_data_set)-1][variable_target] += data_set[i][variable_target] / average_length;  #Contribute to average to latest element of new_data_set
            counter+=1;
        data_set = new_data_set;
        return data_set;
    elif(operation["name"]=="log"):
        """
        . Ln(x) where x is the target variable
        """
        variable_target = operation["variable_target"];
        for i in range(len(data_set)):   #Look through data set, perform operation where required
            data_set[i][variable_target] = math.log(data_set[i][variable_target]);
    elif(operation["name"]=="get_error"):
        variable_target = operation["variable_target"];
        average_length  = operation["average_length"];
        average_set = [];
        for i in range(len(data_set)):
            average_set.append(data_set[i][variable_target]);
            if(i % average_length == (average_length-1)):
                error = np.std( np.array(average_set), ddof=1);    #ddof=1 => standard deviation of the sample
                for j in range(average_length):
                    data_set[i-j].update({"error":error});
                average_set = [];
        return data_set;
    elif(operation["name"]=="get_speedup_efficiency"):
        """
        . Finds the speedup from a set
        . Expects data in the form [[parameters averaged for threads=1], [parameters averaged for threads=N], ...]
            -E.g 1st element is for threads =1 (sequential)

        . NOTE; When considering mixed data, the role of threads & ranks swap (want to see efficiency and speedup for 
            changing ranks, with threads fixed), hence will need to swap out mentions of "threads" for "ranks" specifically
        """
        for i in range(len(data_set)):
            #For MPI, when using ranks>=2, and 0th rank performs no work (just controls data managment)
            #speedup    = data_set[0]["full_time"]/data_set[i]["full_time"];     #Speedup RELATIVE to 0th item's threads -> e.g if starts at 2 ranks, speedup relative to this
            #efficiency = speedup / (data_set[i]["ranks"] -2);

            #When considering 1->N processes, ordered
            speedup      = data_set[i]["full_time"]/data_set[0]["full_time"];               #How many times faster than original; perfect => =thread number
            efficiency   = speedup/data_set[i]["threads"];                      #How well it performed according to number of threads it had
            data_set[i].update({"speedup":speedup});
            data_set[i].update({"efficiency":efficiency});
        return data_set;

def reorder_data(multi_data_set, type):
    """
    . Looks through sets of data given
    . Generates
    """
    combined_data_set = [];
    #Combine all data into 1 long list
    for data_set in multi_data_set:
        for data_parameters in data_set:
            combined_data_set.append(data_parameters);
    #Sort according to some measure
    ordered_multi_data_set = [];
    if(type == "fixParticles_varyThreads"): #Go from particles varied for fixed threads in each document, to this now;
        for data_parameters in combined_data_set:
            #Want to sort for fixed particle number => check if any other lists have your particle number
            sortMatchFound = False;
            isSorted = False;
            for ordered_data_set in ordered_multi_data_set:
                if(ordered_data_set[0]["particles"] == data_parameters["particles"]):
                    #If has same particles, add somewhere in here
                    for ordered_data_parameters_index in range(len(ordered_data_set)):
                        #Look for where to place in here, sorts low to high (in thread numbers)
                        if(data_parameters["threads"] <= ordered_data_set[ordered_data_parameters_index]["threads"]):
                            ordered_data_set.insert(ordered_data_parameters_index, data_parameters);
                            isSorted = True;
                        if(isSorted):
                            break;
                    if(not isSorted):
                        #If searched but couldn't find a lower point, then add to very end of list
                        ordered_data_set.append(data_parameters);
                        isSorted = True;
                    sortMatchFound = True;
                if(isSorted):
                    break;
            if(not sortMatchFound):
                #If could not found another set with particle number like this one, create a new space
                ordered_multi_data_set.append( [data_parameters] );
    return ordered_multi_data_set;

def plot_2D_multi_colours(multi_data_set, data_legend, data_colour, variable_X, variable_Y, showError=True, showLegend=True, plotType="plot", lineType="solid", pointSize=30):
    """
    . Plots 2 parameters against each other from a set of data
    . For each set given, the data is a separate colour

    multi_data_set = List of sets of data, which each hold a set of information about each run of the program
    data_legend = Legend name used for each set of data, e.g. That data set's shared parameter, that is then varied and compared over each separate set
    variable_X = Key of data to use in the X axis
    variable_Y = Key of data to use in the Y axis
    """
    print("Plotting data...");
    for data_set_index in range(len(multi_data_set)):
        #Get each set of data
        legend_label = data_legend[data_set_index];
        colour_label = data_colour[data_set_index];
        x_data = [];
        y_data = [];
        y_error_values = [];
        for data_parameters_index in range(len(multi_data_set[data_set_index])):
            data_parameters = multi_data_set[data_set_index][data_parameters_index];
            try:
                x_data.append(data_parameters[variable_X]);
                y_data.append(data_parameters[variable_Y]);
            except:
                print("--parameter variable could not be found--");
            try:
                error_value = data_parameters["error"];
                y_error_values.append(error_value);
            except:
                y_error_values.append(0.0);
        if(plotType=="plot"):
            if(showLegend):
                plt.plot(x_data, y_data, label=legend_label, color=colour_label, linestyle=lineType);
            else:
                plt.plot(x_data, y_data, color=colour_label, linestyle=lineType);
        elif(plotType=="scatter"):
            if(showLegend):
                plt.scatter(x_data, y_data, label=legend_label, color=colour_label, linestyle=lineType, s=pointSize);
            else:
                plt.scatter(x_data, y_data, color=colour_label, linestyle=lineType, s=pointSize);
        if(showError):
            plt.errorbar(x_data, y_data, yerr=np.array(y_error_values), color=colour_label, linestyle=lineType);

def plot_2D_composite_PRESET(ID):
    """
    . Plots a composite plot using the data you specify
    . This function is changed to fir the type of plot wanted
    . Copies of this function should be made if you would like to revisit a given plot combination

    . Must move ALL data into folder for any graphs to display correctly
    """
    if(ID == "Graph1"):
        #MPI, Both Home and BC4, Euler, ReducedTree

        #HomePc
        avg_multi_data_set = [];
        data_legend = ["HomePc, N=1000", "HomePc, N=1500", "HomePc, N=2000"];
        data_colour = ["red", "green", "blue"];

        raw_data_1000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles1000.txt");
        raw_data_1500 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles1500.txt");
        raw_data_2000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles2000.txt");
        avg_data_1000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles1000.txt");
        avg_data_1500 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles1500.txt");
        avg_data_2000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerReducedTree_particles2000.txt");

        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1500 = operate_on_data_set(avg_data_1500, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_1500 = operate_on_data_set(avg_data_1500, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        avg_multi_data_set.append(avg_data_1000);
        avg_multi_data_set.append(avg_data_1500);
        avg_multi_data_set.append(avg_data_2000);

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        #plot_2D_multi_colours([raw_data_1000, raw_data_1500, raw_data_2000], data_legend, data_colour, "threads", "full_time", showError=False, showLegend=False, plotType="scatter", lineType="solid");

        #BC4
        bc4_avg_multi_data_set = [];
        bc4_data_legend = ["BC4, N=1000", "BC4, N=1500", "BC4, N=2000"];
        bc4_data_colour = ["red", "green", "blue"];
        
        bc4_avg_data_1000 = read_singleInterCalc_data("MPI_BC4_MainOne_Modified_3avg_EulerReducedTree_particles1000.txt");
        bc4_avg_data_1500 = read_singleInterCalc_data("MPI_BC4_MainOne_Modified_3avg_EulerReducedTree_particles1500.txt");
        bc4_avg_data_2000 = read_singleInterCalc_data("MPI_BC4_MainOne_Modified_3avg_EulerReducedTree_particles2000.txt");
        
        bc4_avg_data_1000 = operate_on_data_set(bc4_avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_1500 = operate_on_data_set(bc4_avg_data_1500, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_2000 = operate_on_data_set(bc4_avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_1000 = operate_on_data_set(bc4_avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_1500 = operate_on_data_set(bc4_avg_data_1500, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_2000 = operate_on_data_set(bc4_avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        bc4_avg_multi_data_set.append(bc4_avg_data_1000);
        bc4_avg_multi_data_set.append(bc4_avg_data_1500);
        bc4_avg_multi_data_set.append(bc4_avg_data_2000);

        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");
        #plot_2D_multi_colours([raw_data_1000, raw_data_1500, raw_data_2000], data_legend, data_colour, "threads", "full_time", showError=False, showLegend=False, plotType="scatter", lineType="solid");
        
        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI, ReducedTree, Main Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph2"):
        #MPI, Both Home and BC4, Euler, LinearTree

        #HomePc
        avg_multi_data_set = [];
        data_legend = ["HomePc, N=1000", "HomePc, N=2000"];
        data_colour = ["red", "blue"];

        avg_data_1000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerLinearTree_particles1000.txt");
        avg_data_2000 = read_singleInterCalc_data("MPI_HomePc_MainOne_Modified_3avg_EulerLinearTree_particles2000.txt");

        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        avg_multi_data_set.append(avg_data_1000);
        avg_multi_data_set.append(avg_data_2000);

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        #plot_2D_multi_colours([raw_data_1000, raw_data_1500, raw_data_2000], data_legend, data_colour, "threads", "full_time", showError=False, showLegend=False, plotType="scatter", lineType="solid");

        #BC4
        
        bc4_avg_multi_data_set = [];
        bc4_data_legend = ["BC4, N=1000", "BC4, N=2000"];
        bc4_data_colour = ["red", "blue"];
        
        bc4_avg_data_1000 = read_singleInterCalc_data("MPI_BC4_MainOne_Modified_3avg_EulerLinearTree_particles1000.txt");
        bc4_avg_data_2000 = read_singleInterCalc_data("MPI_BC4_MainOne_Modified_3avg_EulerLinearTree_particles2000.txt");
        
        bc4_avg_data_1000 = operate_on_data_set(bc4_avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_2000 = operate_on_data_set(bc4_avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_1000 = operate_on_data_set(bc4_avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_2000 = operate_on_data_set(bc4_avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        bc4_avg_multi_data_set.append(bc4_avg_data_1000);
        bc4_avg_multi_data_set.append(bc4_avg_data_2000);

        #plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");
        #plot_2D_multi_colours([raw_data_1000, raw_data_1500, raw_data_2000], data_legend, data_colour, "threads", "full_time", showError=False, showLegend=False, plotType="scatter", lineType="solid");
        
        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI, LinearTree, Main Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph3"):
        #OpenMP, HomePc only, linear and reduced tree methods for varying threads, for 2 sets of particles

        #HomePc
        reducedTree_avg_multi_data_set = [];
        avg_multi_data_set             = [];
        data_legend             = ["LinearTree, N=1000", "LinearTree, N=2000"];
        reducedTree_data_legend = ["ReducedTree, N=1000", "ReducedTree, N=2000"];
        data_colour = ["red", "blue"];

        avg_data_1000 = read_singleInterCalc_data("OpenMP_HomePc_MainOne_3avg_EulerLinearTree_particles1000.txt");
        avg_data_2000 = read_singleInterCalc_data("OpenMP_HomePc_MainOne_3avg_EulerLinearTree_particles2000.txt");
        reducedTree_avg_data_1000 = read_singleInterCalc_data("OpenMP_HomePc_MainOne_3avg_EulerReducedTree_particles1000.txt");
        reducedTree_avg_data_2000 = read_singleInterCalc_data("OpenMP_HomePc_MainOne_3avg_EulerReducedTree_particles2000.txt");

        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1000 = operate_on_data_set(avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_2000 = operate_on_data_set(avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        reducedTree_avg_data_1000 = operate_on_data_set(reducedTree_avg_data_1000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        reducedTree_avg_data_2000 = operate_on_data_set(reducedTree_avg_data_2000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        reducedTree_avg_data_1000 = operate_on_data_set(reducedTree_avg_data_1000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        reducedTree_avg_data_2000 = operate_on_data_set(reducedTree_avg_data_2000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        avg_multi_data_set.append(avg_data_1000);
        avg_multi_data_set.append(avg_data_2000);
        reducedTree_avg_multi_data_set.append(reducedTree_avg_data_1000);
        reducedTree_avg_multi_data_set.append(reducedTree_avg_data_2000);

        plot_2D_multi_colours(reducedTree_avg_multi_data_set, reducedTree_data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");
        #plot_2D_multi_colours([raw_data_1000, raw_data_1500, raw_data_2000], data_legend, data_colour, "threads", "full_time", showError=False, showLegend=False, plotType="scatter", lineType="solid");

        plt.xlabel("Number of threads");
        plt.ylabel("Time (s)");
        plt.title("OpenMP, HomePc, Linear and Reduced Tree, Main Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph4"):
        #Plotting reduced tree distribution, found in MPI mode 7, results copied manually
        """
        All have 21 binned spaces

        RANDOM ARRANGEMENT
        0.4 search
        search_range=  1000000000.0 ,  particle=  1000
        frequency_bin=  [  0.   0.  26.  76. 154. 172. 177. 142. 102.  65.  50.  28.   8.   0.   0.   0.   0.   0.   0.   0.   0.]

        0.2 search
        search_range=  500000000.0 ,  particle=  1000
        frequency_bin=  [279. 693.  28.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0    0.   0.   0.   0.   0.   0.   0.]

        0.5 search
        search_range=  1250000000.0 ,  particle=  1000
        frequency_bin=  [  0.   0.   0.   3.  15.  20.  74.  91. 122. 130. 117. 114.  73.  68.  54.  50.  39.  20.   7.   3.   0.]



        DISC ARRANGEMENT
        search_range=  500000000.0 ,  particle=  1000
        frequency_bin=  [  0.  30. 124. 118. 129. 153. 261. 152.  33.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.]

        search_range=  1000000000.0 ,  particle=  1000
        frequency_bin=  [  0.   0.   0.   0.   0.   0.   0.   3.  29.  61.  58.  69.  88.  70.  78.  94. 104. 127. 133.  76.  10.]

        search_range=  1250000000.0 ,  particle=  1000
        frequency_bin=  [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1.  11.  53.  85.  94.  99. 116. 139. 155. 247.]

        """
        particle_number = 1000;
        bin_number = 21;
        bin_size = particle_number / bin_number;

        #Random arrangement
        x_data_0p4 = bin_size*np.array(range(0,bin_number));
        y_data_0p4 = [0,   0,  26,  76, 154, 172, 177, 142, 102,  65,  50,  28,   8,   0,   0,   0,   0,   0,   0,   0,   0];
        plt.plot(x_data_0p4, y_data_0p4, label="Random, search_range= 0.4*L", color="red");

        x_data_0p2 = bin_size*np.array(range(0,bin_number));
        y_data_0p2 = [279, 693,  28,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0];
        plt.plot(x_data_0p2, y_data_0p2, label="Random, search_range= 0.2*L", color="green");

        x_data_0p5 = bin_size*np.array(range(0,bin_number));
        y_data_0p5 = [0,   0,   0,   3,  15,  20,  74,  91, 122, 130, 117, 114,  73,  68,  54,  50,  39,  20,   7,   3,   0];
        plt.plot(x_data_0p5, y_data_0p5, label="Random, search_range= 0.5*L", color="blue");


        #Disc arrangement
        disc_x_data_0p4 = bin_size*np.array(range(0,bin_number));
        disc_y_data_0p4 = [  0,   0,   0,   0,   0,   0,   0,   3,  29,  61,  58,  69,  88,  70,  78,  94, 104, 127, 133,  76,  10]
        plt.plot(disc_x_data_0p4, disc_y_data_0p4, label="Disc, search_range= 0.4*L", color="red", linestyle="dashed");

        disc_x_data_0p2 = bin_size*np.array(range(0,bin_number));
        disc_y_data_0p2 = [  0,  30, 124, 118, 129, 153, 261, 152,  33,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0]
        plt.plot(disc_x_data_0p2, disc_y_data_0p2, label="Disc, search_range= 0.2*L", color="green", linestyle="dashed");

        disc_x_data_0p5 = bin_size*np.array(range(0,bin_number));
        disc_y_data_0p5 = [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  11,  53,  85,  94,  99, 116, 139, 155, 247]
        plt.plot(disc_x_data_0p5, disc_y_data_0p5, label="Disc, search_range= 0.5*L", color="blue", linestyle="dashed");

        plt.xlabel("Binned interaction number");
        plt.ylabel("Frequency");
        plt.title("ReducedTree distribution for 1000 particles, 21 bins");
        plt.legend();
        plt.show();
    if(ID == "Graph5_1"):
        #Plot of MPI HomePc and BC4 for a given N, varying processes
        #This contains the time, speed and efficiency plots fo this data set

        avg_multi_data_set = [];
        data_legend = ["HomePc"];
        data_colour = ["red"];

        avg_data_10000 = read_singleInterCalc_data("MPI_HomePc_MainOne_3avg_PureLinear_particles10000.txt");
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"get_speedup_efficiency"});
        avg_multi_data_set.append(avg_data_10000);

        #BC4Time (s)
        bc4_avg_multi_data_set = [];
        bc4_data_legend = ["BC4"];
        bc4_data_colour = ["blue"];

        bc4_avg_data_10000 = read_singleInterCalc_data("MPI_BC4_MainOne_3avg_PureLinear_particles10000.txt");
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"get_speedup_efficiency"});
        bc4_avg_multi_data_set.append(bc4_avg_data_10000);

        #Plots
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "speedup", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "speedup", showError=False, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Speedup");
        plt.title("MPI, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();
    
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "efficiency", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "efficiency", showError=False, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Efficiency");
        plt.title("MPI, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();
    if(ID == "Graph5_2"):
        avg_multi_data_set = [];
        data_legend = ["HomePc"];
        data_colour = ["red"];

        avg_data_10000 = read_singleInterCalc_data("OpenMP_HomePc_MainOne_3avg_PureLinear_particles10000.txt");
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_10000 = operate_on_data_set(avg_data_10000, {"name":"get_speedup_efficiency"});
        avg_multi_data_set.append(avg_data_10000);

        #BC4Time (s)
        bc4_avg_multi_data_set = [];
        bc4_data_legend = ["BC4"];
        bc4_data_colour = ["blue"];

        bc4_avg_data_10000 = read_singleInterCalc_data("OpenMP_BC4_MainOne_3avg_PureLinear_particles10000.txt");
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        bc4_avg_data_10000 = operate_on_data_set(bc4_avg_data_10000, {"name":"get_speedup_efficiency"});
        bc4_avg_multi_data_set.append(bc4_avg_data_10000);

        #Plots
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("OpenMP, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "speedup", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "speedup", showError=False, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Speedup");
        plt.title("OpenMP, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();
    
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "threads", "efficiency", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plot_2D_multi_colours(bc4_avg_multi_data_set, bc4_data_legend, bc4_data_colour, "threads", "efficiency", showError=False, showLegend=True, plotType="plot", lineType="dashed");
        plt.xlabel("Number of ranks");
        plt.ylabel("Efficiency");
        plt.title("OpenMP, PureLinear, Main Calculation, N=10000");
        plt.legend();
        plt.show();
    if(ID == "Graph6_1"):
        #MPI and OpenMP data from BC4 on single calcualtion times (single interaction set) for varying particle numbers
        #This plots RANKS vs TIME

        bc4_data_colour = get_colour_gradient(5);
        #MPI
        MPI_bc4_avg_multi_data_set = [];
        MPI_bc4_data_legend = ["MPI, N=5000", "MPI, N=25000", "MPI, N=35000", "MPI, N=45000", "MPI, N=55000"];  #, "MPI, N=15000"

        MPI_bc4_avg_data_5000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles5000.txt");
        MPI_bc4_avg_data_5000 = operate_on_data_set(MPI_bc4_avg_data_5000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_5000 = operate_on_data_set(MPI_bc4_avg_data_5000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_5000);
        #MPI_bc4_avg_data_15000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles15000.txt");
        #MPI_bc4_avg_data_15000 = operate_on_data_set(MPI_bc4_avg_data_15000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        #MPI_bc4_avg_data_15000 = operate_on_data_set(MPI_bc4_avg_data_15000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        #MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_15000);
        MPI_bc4_avg_data_25000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles25000.txt");
        MPI_bc4_avg_data_25000 = operate_on_data_set(MPI_bc4_avg_data_25000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_25000 = operate_on_data_set(MPI_bc4_avg_data_25000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_25000);
        MPI_bc4_avg_data_35000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles35000.txt");
        MPI_bc4_avg_data_35000 = operate_on_data_set(MPI_bc4_avg_data_35000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_35000 = operate_on_data_set(MPI_bc4_avg_data_35000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_35000);
        MPI_bc4_avg_data_45000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles45000.txt");
        MPI_bc4_avg_data_45000 = operate_on_data_set(MPI_bc4_avg_data_45000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_45000 = operate_on_data_set(MPI_bc4_avg_data_45000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_45000);
        MPI_bc4_avg_data_55000 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_particles55000.txt");
        MPI_bc4_avg_data_55000 = operate_on_data_set(MPI_bc4_avg_data_55000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_55000 = operate_on_data_set(MPI_bc4_avg_data_55000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_55000);

        plot_2D_multi_colours(MPI_bc4_avg_multi_data_set, MPI_bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        
        plt.xlabel("Number of processes");
        plt.ylabel("Time (s)");
        plt.title("BC4, Single Interaction Set Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph6_2"):
        #MPI and OpenMP data from BC4 on single calcualtion times (single interaction set) for varying particle numbers
        #This plots PARTICLES vs TIME

        bc4_data_colour = get_colour_gradient(5);
        #MPI
        MPI_bc4_avg_multi_data_set = [];
        MPI_bc4_data_legend = ["MPI, ranks=2", "MPI, ranks=4", "MPI, ranks=8", "MPI, ranks=16", "MPI, ranks=24"];

        MPI_bc4_avg_data_2 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_ranks2.txt");
        MPI_bc4_avg_data_2 = operate_on_data_set(MPI_bc4_avg_data_2, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_2 = operate_on_data_set(MPI_bc4_avg_data_2, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_2);
        MPI_bc4_avg_data_4 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_ranks4.txt");
        MPI_bc4_avg_data_4 = operate_on_data_set(MPI_bc4_avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_4 = operate_on_data_set(MPI_bc4_avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_4);
        MPI_bc4_avg_data_8 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_ranks8.txt");
        MPI_bc4_avg_data_8 = operate_on_data_set(MPI_bc4_avg_data_8, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_8 = operate_on_data_set(MPI_bc4_avg_data_8, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_8);
        MPI_bc4_avg_data_16 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_ranks16.txt");
        MPI_bc4_avg_data_16 = operate_on_data_set(MPI_bc4_avg_data_16, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_16 = operate_on_data_set(MPI_bc4_avg_data_16, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_16);
        MPI_bc4_avg_data_24 = read_singleInterCalc_data("MPI_BC4_SingleCalc_5avg_ranks24.txt");
        MPI_bc4_avg_data_24 = operate_on_data_set(MPI_bc4_avg_data_24, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_data_24 = operate_on_data_set(MPI_bc4_avg_data_24, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        MPI_bc4_avg_multi_data_set.append(MPI_bc4_avg_data_24);

        plot_2D_multi_colours(MPI_bc4_avg_multi_data_set, MPI_bc4_data_legend, bc4_data_colour, "particles", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        
        
        plt.xlabel("Number of particles");
        plt.ylabel("Time (s)");
        plt.title("BC4, Single Interaction Set Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph6_3"):
        bc4_data_colour = get_colour_gradient(5);
        #OpenMP
        OpenMP_bc4_avg_multi_data_set = [];
        OpenMP_bc4_data_legend = ["OpenMP, N=5000", "OpenMP, N=25000", "OpenMP, N=35000", "OpenMP, N=45000", "OpenMP, N=55000"];  #, "OpenMP, N=15000"

        OpenMP_bc4_avg_data_5000 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_particles5000.txt");
        OpenMP_bc4_avg_data_5000 = operate_on_data_set(OpenMP_bc4_avg_data_5000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_5000 = operate_on_data_set(OpenMP_bc4_avg_data_5000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_5000);
        OpenMP_bc4_avg_data_25000 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_particles25000.txt");
        OpenMP_bc4_avg_data_25000 = operate_on_data_set(OpenMP_bc4_avg_data_25000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_25000 = operate_on_data_set(OpenMP_bc4_avg_data_25000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_25000);
        OpenMP_bc4_avg_data_35000 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_particles35000.txt");
        OpenMP_bc4_avg_data_35000 = operate_on_data_set(OpenMP_bc4_avg_data_35000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_35000 = operate_on_data_set(OpenMP_bc4_avg_data_35000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_35000);
        OpenMP_bc4_avg_data_45000 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_particles45000.txt");
        OpenMP_bc4_avg_data_45000 = operate_on_data_set(OpenMP_bc4_avg_data_45000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_45000 = operate_on_data_set(OpenMP_bc4_avg_data_45000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_45000);
        OpenMP_bc4_avg_data_55000 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_particles55000.txt");
        OpenMP_bc4_avg_data_55000 = operate_on_data_set(OpenMP_bc4_avg_data_55000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_55000 = operate_on_data_set(OpenMP_bc4_avg_data_55000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_55000);

        plot_2D_multi_colours(OpenMP_bc4_avg_multi_data_set, OpenMP_bc4_data_legend, bc4_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");

        plt.xlabel("Number of processes");
        plt.ylabel("Time (s)");
        plt.title("BC4, Single Interaction Set Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph6_4"):
        #OpenMP switch plots of 6_3 -> vary threads

        bc4_data_colour = get_colour_gradient(5);
        #OpenMP
        OpenMP_bc4_avg_multi_data_set = [];
        OpenMP_bc4_data_legend = ["OpenMP, ranks=2", "OpenMP, ranks=4", "OpenMP, ranks=8", "OpenMP, ranks=16", "OpenMP, ranks=24"];

        OpenMP_bc4_avg_data_2 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_ranks2.txt");
        OpenMP_bc4_avg_data_2 = operate_on_data_set(OpenMP_bc4_avg_data_2, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_2 = operate_on_data_set(OpenMP_bc4_avg_data_2, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_2);
        OpenMP_bc4_avg_data_4 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_ranks4.txt");
        OpenMP_bc4_avg_data_4 = operate_on_data_set(OpenMP_bc4_avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_4 = operate_on_data_set(OpenMP_bc4_avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_4);
        OpenMP_bc4_avg_data_8 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_ranks8.txt");
        OpenMP_bc4_avg_data_8 = operate_on_data_set(OpenMP_bc4_avg_data_8, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_8 = operate_on_data_set(OpenMP_bc4_avg_data_8, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_8);
        OpenMP_bc4_avg_data_16 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_ranks16.txt");
        OpenMP_bc4_avg_data_16 = operate_on_data_set(OpenMP_bc4_avg_data_16, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_16 = operate_on_data_set(OpenMP_bc4_avg_data_16, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_16);
        OpenMP_bc4_avg_data_24 = read_singleInterCalc_data("OpenMP_BC4_SingleCalc_5avg_ranks24.txt");
        OpenMP_bc4_avg_data_24 = operate_on_data_set(OpenMP_bc4_avg_data_24, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_data_24 = operate_on_data_set(OpenMP_bc4_avg_data_24, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        OpenMP_bc4_avg_multi_data_set.append(OpenMP_bc4_avg_data_24);

        plot_2D_multi_colours(OpenMP_bc4_avg_multi_data_set, OpenMP_bc4_data_legend, bc4_data_colour, "particles", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        
        
        plt.xlabel("Number of particles");
        plt.ylabel("Time (s)");
        plt.title("BC4, Single Interaction Set Calculation");
        plt.legend();
        plt.show();


    if(ID == "Graph7_1"):
        #MPI, Mixed method(MPI & OpenMP), for reduced tree only, ranksVsTime for varying threads
        #This is done on BC4

        #1000 particles
        avg_multi_data_set = [];
        data_legend = ["Threads=1, N=1000", "Threads=4, N=1000"];
        data_colour = ["red", "blue"];

        avg_data_1 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads1_particles1000.txt");
        avg_data_4 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads4_particles1000.txt");

        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        avg_multi_data_set.append(avg_data_1);
        avg_multi_data_set.append(avg_data_4);

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");

        #3000 particles
        N3000_avg_multi_data_set = [];
        N3000_data_legend = ["Threads=1, N=3000", "Threads=4, N=3000"];
        N3000_data_colour = ["red", "blue"];
        
        N3000_avg_data_1 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads1_particles3000.txt");
        N3000_avg_data_4 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads4_particles3000.txt");

        N3000_avg_data_1 = operate_on_data_set(N3000_avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N3000_avg_data_4 = operate_on_data_set(N3000_avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N3000_avg_data_1 = operate_on_data_set(N3000_avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        N3000_avg_data_4 = operate_on_data_set(N3000_avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        N3000_avg_multi_data_set.append(N3000_avg_data_1);
        N3000_avg_multi_data_set.append(N3000_avg_data_4);

        plot_2D_multi_colours(N3000_avg_multi_data_set, N3000_data_legend, N3000_data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");


        #5000 particles
        N5000_avg_multi_data_set = [];
        N5000_data_legend = ["Threads=1, N=5000", "Threads=4, N=5000"];
        N5000_data_colour = ["red", "blue"];
        
        N5000_avg_data_1 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads1_particles5000.txt");
        N5000_avg_data_4 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_threads4_particles5000.txt");

        N5000_avg_data_1 = operate_on_data_set(N5000_avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_4 = operate_on_data_set(N5000_avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_1 = operate_on_data_set(N5000_avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_4 = operate_on_data_set(N5000_avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        N5000_avg_multi_data_set.append(N5000_avg_data_1);
        N5000_avg_multi_data_set.append(N5000_avg_data_4);

        plot_2D_multi_colours(N5000_avg_multi_data_set, N5000_data_legend, N5000_data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dotted");


        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI and OpenMp Mixed Approach, BC4, ReducedTree");
        plt.legend();
        plt.show();
    if(ID == "Graph7_2"):
        #MPI, Mixed method(MPI & OpenMP), for reduced tree only, ranksVsTime for varying threads
        #This is done om HomePc

        #1000 particles
        """
        avg_multi_data_set = [];
        data_legend = ["Threads=1, N=1000", "Threads=4, N=1000"];
        data_colour = ["red", "blue"];

        avg_data_1 = read_singleInterCalc_data("MPI_HomePc_Mixed_3avg_threads1_particles1000.txt");
        avg_data_4 = read_singleInterCalc_data("MPI_HomePc_Mixed_3avg_threads4_particles1000.txt");

        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        avg_multi_data_set.append(avg_data_1);
        avg_multi_data_set.append(avg_data_4);

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");

        """
        #5000 particles
        N5000_avg_multi_data_set = [];
        N5000_data_legend = ["Threads=1, N=5000", "Threads=4, N=5000"];
        N5000_data_colour = ["red", "blue"];
        
        N5000_avg_data_1 = read_singleInterCalc_data("MPI_HomePc_Mixed_3avg_threads1_particles5000.txt");
        N5000_avg_data_4 = read_singleInterCalc_data("MPI_HomePc_Mixed_3avg_threads4_particles5000.txt");

        N5000_avg_data_1 = operate_on_data_set(N5000_avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_4 = operate_on_data_set(N5000_avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_1 = operate_on_data_set(N5000_avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        N5000_avg_data_4 = operate_on_data_set(N5000_avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});

        N5000_avg_multi_data_set.append(N5000_avg_data_1);
        N5000_avg_multi_data_set.append(N5000_avg_data_4);

        plot_2D_multi_colours(N5000_avg_multi_data_set, N5000_data_legend, N5000_data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");


        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI and OpenMp Mixed Approach, HomePc, ReducedTree");
        plt.legend();
        plt.show();
    if(ID == "Graph8"):
        #Speedup and efficiency of the N=1000 BC4 calcualtion using the mixed approach

        avg_multi_data_set = [];
        data_legend = ["Threads=1", "Threads=4"];
        data_colour = ["red", "blue"];

        avg_data_1 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_1threads_particles1000_excluded.txt");
        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_1 = operate_on_data_set(avg_data_1, {"name":"get_speedup_efficiency"});
        avg_multi_data_set.append(avg_data_1);

        avg_data_4 = read_singleInterCalc_data("MPI_BC4_Mixed_3avg_4threads_particles1000_excluded.txt");
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"get_error", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"average_ordered", "variable_target":"full_time", "average_length":3});
        avg_data_4 = operate_on_data_set(avg_data_4, {"name":"get_speedup_efficiency"});
        avg_multi_data_set.append(avg_data_4);

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "ranks", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        plt.xlabel("Number of ranks");
        plt.ylabel("Time (s)");
        plt.title("MPI and OpenMp Mixed Approach, BC4, ReducedTree, N=1000");
        plt.legend();
        plt.show();
        
        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "ranks", "speedup", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plt.xlabel("Number of ranks");
        plt.ylabel("Speedup");
        plt.title("MPI and OpenMp Mixed Approach, BC4, ReducedTree, N=1000");
        plt.legend();
        plt.show();

        plot_2D_multi_colours(avg_multi_data_set, data_legend, data_colour, "ranks", "efficiency", showError=False, showLegend=True, plotType="plot", lineType="solid");
        plt.xlabel("Number of ranks");
        plt.ylabel("Efficiency");
        plt.title("MPI and OpenMp Mixed Approach, BC4, ReducedTree, N=1000");
        plt.legend();
        plt.show();
    if(ID == "Graph9"):
        #OpenMP singleCalc graphs

        #HomePc, Varying particles
        home_avg_multi_data_set = [];
        home_data_legend = ["OpenMP, N=5000", "OpenMP, N=25000", "OpenMP, N=35000", "OpenMP, N=45000", "OpenMP, N=55000"];
        home_data_colour = get_colour_gradient(5);

        avg_data_home5000  = read_singleInterCalc_data("OpenMP_home_singleCalc_5avg_particles5000.txt");
        avg_data_home25000 = read_singleInterCalc_data("OpenMP_home_singleCalc_5avg_particles25000.txt");
        avg_data_home35000 = read_singleInterCalc_data("OpenMP_home_singleCalc_5avg_particles35000.txt");
        avg_data_home45000 = read_singleInterCalc_data("OpenMP_home_singleCalc_5avg_particles45000.txt");
        avg_data_home55000 = read_singleInterCalc_data("OpenMP_home_singleCalc_5avg_particles55000.txt");

        avg_data_home5000  = operate_on_data_set(avg_data_home5000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home5000  = operate_on_data_set(avg_data_home5000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        avg_data_home25000 = operate_on_data_set(avg_data_home25000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home25000 = operate_on_data_set(avg_data_home25000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        avg_data_home35000 = operate_on_data_set(avg_data_home35000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home35000 = operate_on_data_set(avg_data_home35000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        avg_data_home45000 = operate_on_data_set(avg_data_home45000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home45000 = operate_on_data_set(avg_data_home45000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});
        avg_data_home55000 = operate_on_data_set(avg_data_home55000, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home55000 = operate_on_data_set(avg_data_home55000, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        home_avg_multi_data_set.append(avg_data_home5000);
        home_avg_multi_data_set.append(avg_data_home25000);
        home_avg_multi_data_set.append(avg_data_home35000);
        home_avg_multi_data_set.append(avg_data_home45000);
        home_avg_multi_data_set.append(avg_data_home55000);

        plot_2D_multi_colours(home_avg_multi_data_set, home_data_legend, home_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");
        plt.xlabel("Number of threads");
        plt.ylabel("Time (s)");
        plt.title("OpenMP, HomePc, Single Interaction Set Calculation");
        plt.legend();
        plt.show();
    if(ID == "Graph12"):
        #OpenMP singleCalc schedule vary

        #HomePc, Varying particles
        home_avg_multi_data_set = [];
        home_data_legend = ["schedule='static', N=55000", "schedule='dynamic', N=55000", "schedule='guided', N=55000"];
        home_data_colour = ["red", "green", "blue"];

        avg_data_home55000_static = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles55000_staticSchedule.txt");
        avg_data_home55000_static = operate_on_data_set(avg_data_home55000_static, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home55000_static = operate_on_data_set(avg_data_home55000_static, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        avg_data_home55000_dynamic = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles55000_dynamicSchedule.txt");
        avg_data_home55000_dynamic = operate_on_data_set(avg_data_home55000_dynamic, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home55000_dynamic = operate_on_data_set(avg_data_home55000_dynamic, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        avg_data_home55000_guided  = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles55000_guidedSchedule.txt");
        avg_data_home55000_guided = operate_on_data_set(avg_data_home55000_guided, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home55000_guided = operate_on_data_set(avg_data_home55000_guided, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        home_avg_multi_data_set.append(avg_data_home55000_static);
        home_avg_multi_data_set.append(avg_data_home55000_dynamic);
        home_avg_multi_data_set.append(avg_data_home55000_guided);

        plot_2D_multi_colours(home_avg_multi_data_set, home_data_legend, home_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="solid");



        home_avg_multi_data_set = [];
        home_data_legend = ["schedule='static', N=110000", "schedule='dynamic', N=110000", "schedule='guided', N=110000"];
        home_data_colour = ["red", "green", "blue"];

        avg_data_home110000_static = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles110000_staticSchedule.txt");
        avg_data_home110000_static = operate_on_data_set(avg_data_home110000_static, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home110000_static = operate_on_data_set(avg_data_home110000_static, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        avg_data_home110000_dynamic = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles110000_dynamicSchedule.txt");
        avg_data_home110000_dynamic = operate_on_data_set(avg_data_home110000_dynamic, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home110000_dynamic = operate_on_data_set(avg_data_home110000_dynamic, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        avg_data_home110000_guided  = read_singleInterCalc_data("OpenMP_HomePc_SingleCalc_5avg_particles110000_guidedSchedule.txt");
        avg_data_home110000_guided = operate_on_data_set(avg_data_home110000_guided, {"name":"get_error", "variable_target":"full_time", "average_length":5});
        avg_data_home110000_guided = operate_on_data_set(avg_data_home110000_guided, {"name":"average_ordered", "variable_target":"full_time", "average_length":5});

        home_avg_multi_data_set.append(avg_data_home110000_static);
        home_avg_multi_data_set.append(avg_data_home110000_dynamic);
        home_avg_multi_data_set.append(avg_data_home110000_guided);

        plot_2D_multi_colours(home_avg_multi_data_set, home_data_legend, home_data_colour, "threads", "full_time", showError=True, showLegend=True, plotType="plot", lineType="dashed");


        plt.xlabel("Number of threads");
        plt.ylabel("Time (s)");
        plt.title("OpenMP, HomePc, Single Interaction Set Calculation");
        plt.legend();
        plt.show();


plot_2D_composite_PRESET("Graph8");