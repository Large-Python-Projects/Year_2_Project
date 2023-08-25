import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import cumtrapz
from scipy.constants import pi, mu_0, hbar

def absorption_arr_integrate(magnetic_field_arr, freq_arr, diff_absorp_arr):
    """
    This function takes the absorption data and integrates it with respect to the magnetic field values.
    ----------
    PARAMETERS
    ----------
    magnetic_field_arr, freq_arr, diff_absorp_arr
    -------
    RETURNS
    -------
    x_mid: A 1D containing the average values between each trapezium in the integral
    new_absorptionarr: A 2D array - The integrated absorption data array
    """
    new_absorptionarr = np.array([])
    
    for row in range(0, np.size(diff_absorp_arr,0)):
        new_absorptionarr_integral = cumtrapz(diff_absorp_arr[row,:], magnetic_field_arr) #integrates the differential absorption data using the trapezium rule
        
        if min(new_absorptionarr_integral) < 0:
            new_absorptionarr_integral += -min(new_absorptionarr_integral) #standardizes the data so it contains no negative values

        if row == 0:
            new_absorptionarr = new_absorptionarr_integral #Because vstack can't stack rows on empty arrays
            
        else:        
            new_absorptionarr = np.vstack((new_absorptionarr, new_absorptionarr_integral))
    
    x_mid=0.5*(magnetic_field_arr[:-1] + magnetic_field_arr[1:])

    return x_mid, new_absorptionarr
    
def Lorentz_peak_func(H_mid, I_0, delta_H, H_0):
    """
    Model function containing the Lorentz Peak equation.
    ----------
    PARAMETERS
    ----------
    H_mid, I_0, delta_H, H_0 are all fitting parameters
    -------
    RETURNS
    -------
    (I_0/(2*pi))*(delta_H/(((H_mid - H_0)**2)+((delta_H/2)**2))): The Lorentz Peak equation
    """
    return (I_0/(2*pi))*(delta_H/(((H_mid - H_0)**2)+((delta_H/2)**2)))
        
def unpack_and_calc_parameters(x_mid, new_absorptionarr):
    """
    Unpacks and processes parameters into appropriate arrays for use throughout the code
    ----------
    PARAMETERS
    ----------
    x_mid, new_absorptionarr
    -------
    RETURNS
    -------
    peak_vals_arr: A 2D array containing the peak values for each frequency
    lorentz_parameters_arr: A 2D array containing the fitting parameters that correspond to the lorentz function for each frequency
    lorentz_error_arr: A 2D array containing all of the errors of the lorentz fitting parameters for each frequency
    peak_width_arr: A 1D array containing all of the peak widths for each frequency
    peak_width_error_arr: A 1D array containing all of errors in the peak widths for each frequency
    peak_position_arr: A 1D array containing all of the peak positions for each frequency
    """
    lorentz_parameters_arr = np.array([])
    peak_vals_arr = np.array([])
    lorentz_error_arr = np.array([])
    
    for row in range(0, np.size(new_absorptionarr,0)):
        lorentz_parameters,pcov=curve_fit(Lorentz_peak_func,x_mid,new_absorptionarr[row,:], p0 = [1,1,10000]) #returns fitting parameters and variance co-variance matrix. Initial guess means all functions are plotted correctly.
        lorentz_error=np.sqrt(np.diag(pcov)) #takes the square root of the diagonal elements in the variance co-variance matrix. This is the error in the fitting parameters
        
        max_func = lambda func: -Lorentz_peak_func(func, *lorentz_parameters) 
        
        peak_vals = minimize_scalar(max_func, bounds=(x_mid.min(),x_mid.max())) #Optimises to find the peak parameters
        peak_vals_arr = np.append(peak_vals_arr, peak_vals['x']) #creates array of peak_vals for ease of use
        
        if row == 0:
            lorentz_parameters_arr = lorentz_parameters #cant use np.vstack on empty arrays
            lorentz_error_arr = lorentz_error
            
        else: 
            lorentz_parameters_arr = np.vstack((lorentz_parameters_arr, lorentz_parameters)) #adds each row into a another array, creating a 2D array
            lorentz_error_arr = np.vstack((lorentz_error_arr, lorentz_error))

    peak_width_arr = lorentz_parameters_arr[:,1] #extracts parameters and returns them for ease of use throughout the code
    peak_width_error_arr = lorentz_error_arr[:,1]
    peak_position_arr = lorentz_parameters_arr[:,2]
    
    return  peak_vals_arr, lorentz_parameters_arr, lorentz_error_arr, peak_width_arr, peak_width_error_arr, peak_position_arr
 
def plot_absorption_20GHz(peak_vals_arr, lorentz_parameters_arr, x_mid, magnetic_field_arr, new_absorptionarr, freq_arr):
    """
    Plots absorption against magnetic field for a frequency of 20GHz
    ----------
    PARAMETERS
    ----------
    peak_vals_arr, lorentz_parameters_arr, x_mid, freq_arr   
    -------
    RETURNS
    -------
    None
    """
    try:
        peak_val_20GHz = peak_vals_arr[freq_arr.index(20.00)] #extracts parameters corresponding to 20GHz 
        lorentz_parameters = lorentz_parameters_arr[freq_arr.index(20.00), :]
                                                    
    except RuntimeError:
        raise RuntimeError('No 20GHz data in the file or 20GHz data couldnt be identified')
        
    x_fit=np.linspace(x_mid.min(),x_mid.max(),500) #generates values for the fit in the appropriate x range
    y_fit=Lorentz_peak_func(x_fit, *lorentz_parameters) #puts the values into the lorentz peak equaton for a fit
    
    plt.figure('py16jo - Absorption vs Magnetic field at 20GHz - Peak')
    plt.title('py16jo - Absorption vs Magnetic field at 20GHz - Peak')
    plt.xlabel('Magnetic Field (A/M)')
    plt.ylabel('Absorption (arb units)')
    
    plt.plot(x_mid, new_absorptionarr[freq_arr.index(20.00), :], 'b.')
    plt.plot(x_fit, y_fit, 'r-')
    plt.plot(peak_val_20GHz, Lorentz_peak_func(peak_val_20GHz, *lorentz_parameters), 'ko') #plots graphs on the same figure for comparison

def kittel_equ(H, gamma, H_k, M_s):
    """
    Model function containing the Kittel equation.
    ----------
    PARAMETERS
    ----------
    H, gamma, H_k, M_s are all fitting parameters
    -------
    RETURNS
    -------
    ((mu_0*gamma)/2*(pi))*(np.sqrt((H + H_k)*(H + H_k + M_s))): The Kittel equation
    """
    return ((mu_0*gamma)/2*(pi))*(np.sqrt((H + H_k)*(H + H_k + M_s)))

def plot_freq_vs_peak_pos_and_fit(peak_position_arr, freq_arr, lorentz_error_arr):
    """
    Plots frequency against peak positions and fits it using curve_fit for all frequencies
    ----------
    PARAMETERS
    ----------
    peak_position_arr, freq_arr, lorentz_error_arr 
    -------
    RETURNS
    -------
    kittel_parameters: A 2D array containing the fitting parameters that correspond to the kittel equation for each frequency
    kittel_error: A 2D array containing all of the errors of the kittel fitting parameters for each frequency
    """
    kittel_parameters,pcov=curve_fit(kittel_equ, peak_position_arr, freq_arr)
    kittel_error=np.sqrt(np.diag(pcov))
    x_fit=np.linspace(peak_position_arr.min(),peak_position_arr.max(),1000) 
    y_fit=kittel_equ(x_fit, *kittel_parameters)
    
    plt.figure('py16jo - Frequency vs Peak position')
    plt.title('py16jo - Frequency vs Peak position')
    plt.xlabel('Magnetic Field (A/M)')
    plt.ylabel('Frequency (GHz)')
    plt.errorbar(peak_position_arr,freq_arr,xerr=lorentz_error_arr[:,2],capsize=3,fmt='ro',label='data') #sets up the error bars for the graph
    plt.plot(peak_position_arr, freq_arr, 'r.')
    plt.plot(x_fit, y_fit, 'g-')
    plt.show()
    
    return kittel_parameters, kittel_error

def fitting_ratio_equ(freq, delta_H_0, alpha_gamma):
    """
    Model function containing the fitting ratio equation.
    ----------
    PARAMETERS
    ----------
    freq, delta_H_0, alpha, gamma are all fitting parameters
    -------
    RETURNS
    -------
    (delta_H_0 + ((alpha*4*pi)/(mu_0*math.sqrt(3)))*(freq/gamma)): The fitting ratio equation
    """
    freq = np.array(freq) #so that it can be multiplied by a float
    return (delta_H_0 + ((alpha_gamma*4*pi)/(mu_0*np.sqrt(3)))*(freq))
    
def peak_width_vs_freq_and_fit(peak_width_arr, freq_arr, peak_width_error_arr, kittel_parameters):
    
    fitting_ratio_parameters,pcov=curve_fit(fitting_ratio_equ, freq_arr, peak_width_arr)
    fitting_ratio_error=np.sqrt(np.diag(pcov))
    y_fit=fitting_ratio_equ(freq_arr, *fitting_ratio_parameters)
    
    plt.figure('py16jo - Peak width vs Frequency')
    plt.title('py16jo - Peak width vs Frequency')
    plt.ylabel('Peak width (A/M)')
    plt.xlabel('Frequency (GHz)')
    plt.xlim(0,max(freq_arr)+5) #for appropriate x axis range
    plt.errorbar(freq_arr, peak_width_arr, yerr=peak_width_error_arr, capsize=3, fmt='ro',label='data') #generates error bars for the plotted data
    plt.plot(freq_arr, y_fit, 'g-')
    plt.plot(freq_arr, peak_width_arr, 'b.')
    plt.show()
    
    return fitting_ratio_parameters, fitting_ratio_error

def work_out_g_factor(kittel_parameters, kittel_error, fitting_ratio_parameters, fitting_ratio_error):
    """
    Works out the g factor and its corresponding before appending both values to kittel_parameters
    and kittel_error accordingly. These are then renamed and returned as below.
    ----------
    PARAMETERS
    ----------
    kittel_parameters, kittel_error
    -------
    RETURNS
    -------
    final_values_arr: Contains final values for gamma, Hk, Ms and g
    final_values_error_arr: Contains final values for error in gamma, Hk, Ms and g
    """
    mu_b =  9.27400968e-24 #value for the Bohr Magneton
    
    gamma = kittel_parameters[0] #retrieves gamma and its uncertainty to work out g and g_error
    gamma_uncert = kittel_error[0]
    alpha_over_gamma = fitting_ratio_parameters[1]
    fitting_ratio_parameters[1] = fitting_ratio_parameters[1]*gamma
    fitting_ratio_error[1] = fitting_ratio_parameters[1]*(np.sqrt((((fitting_ratio_error[1]/alpha_over_gamma)**2)-((gamma_uncert/gamma)**2))))
    
    g_factor = (gamma*hbar)/mu_b
    g_factor_uncert = (gamma_uncert*hbar)/mu_b

    final_values_arr = np.append(kittel_parameters, g_factor) #appends them in an array and returns the array (rather than each individual value)
    final_values_error_arr = np.append(kittel_error, g_factor_uncert)
    
    return final_values_arr, final_values_error_arr, fitting_ratio_parameters, fitting_ratio_error
    
def val_dictionary(peak_vals_arr, lorentz_error_arr, freq_arr, final_values_arr, final_values_error_arr, peak_width_arr, peak_width_error_arr, fitting_ratio_parameters, fitting_ratio_error):
    """
    Assigns the appropriate keys to the appropriate values for ease of use.
    ----------
    PARAMETERS
    ----------
    peak_vals_arr, lorentz_error_arr, freq_arr, final_values_arr, 
    final_values_error_arr, peak_width_arr, peak_width_error_arr, 
    fitting_ratio_parameters, fitting_ratio_error
    -------
    RETURNS
    -------
    results: A dictionary containing the final calculated values
    """
    results = dict()
    
    results["20GHz_peak"] = peak_vals_arr[freq_arr.index(20.00)]
    results["20GHz_peak_error"] = lorentz_error_arr[freq_arr.index(20.00), 2]
                                         
    results["20GHz_width"] = peak_width_arr[freq_arr.index(20.00)]
    results["20GHz_width_error"] = peak_width_error_arr[freq_arr.index(20.00)]
                                             
    results["DeltaH"] = fitting_ratio_parameters[0]
    results["DeltaH_error"] = fitting_ratio_error[0]

    results["alpha"] = fitting_ratio_parameters[1]
    results["alpha_error"] = fitting_ratio_error[1]

    results["gamma"] = final_values_arr[0]
    results["gamma_error"] = final_values_error_arr[0]

    results["Hk"] = final_values_arr[1]
    results["Hk_error"] = final_values_error_arr[1]

    results["Ms"] = final_values_arr[2]
    results["Ms_error"] = final_values_error_arr[2]

    results["g"] = final_values_arr[3]
    results["g_error"] = final_values_error_arr[3]

    return results
                                      
def ProcessData(filename):
    """
    Opens the data and processes it so it can be used throughout the rest of the code. Also calls each function when necessary.
    ----------
    PARAMETERS
    ----------
    filename: The name of the file to be opened
    -------
    RETURNS
    -------
    freq_arr: An array of each frequency value in order
    magnetic_field_arr: An array of each magnetic field strength value in order 
    diff_absorp_arr: Array of the differential of the absorption data
    results: A dictionary containing the final calculated values
    """
    freq_arr = []
    with open(filename, 'r') as raw_data:
        for n,line in enumerate(raw_data): #counts the lines read for use in np.loadtxt
            line = line.strip()
            if 'Magnetic Field (A/M)' in line:
                dat = line.split()
                for freq in dat:
                    if 'GHz' in freq:
                        freq = float(freq.replace("GHz", "")) #removes GHz from frequencies for ease of use
                        freq_arr.append(freq)
                    else:
                        pass
                break
        
    try:
        data=np.loadtxt(filename, delimiter='\t', skiprows = n+1, unpack = True, dtype = float) #opens the file in one, skipping the metadata and frequency rows, and puts it all in 2D array - data
        
    except IOError:
        raise IOError('File could not be opened')
        
    magnetic_field_arr = data[0,:]                       #extracts magnetic field strength values from data and forms its own 1D array
    diff_absorp_arr = np.delete(data,(0), axis=0)        #deletes magnetic field strength values from data and forms 2D array of differential absorption data
    
    x_mid, new_absorptionarr = absorption_arr_integrate(magnetic_field_arr, freq_arr, diff_absorp_arr)
    
    peak_vals_arr, lorentz_parameters_arr, lorentz_error_arr, peak_width_arr, peak_width_error_arr, peak_position_arr = unpack_and_calc_parameters(x_mid, new_absorptionarr)
    
    plot_absorption_20GHz(peak_vals_arr, lorentz_parameters_arr, x_mid, magnetic_field_arr, new_absorptionarr, freq_arr)
    
    kittel_parameters, kittel_error = plot_freq_vs_peak_pos_and_fit(peak_position_arr, freq_arr, lorentz_error_arr)
    
    fitting_ratio_parameters, fitting_ratio_error = peak_width_vs_freq_and_fit(peak_width_arr, freq_arr, peak_width_error_arr, kittel_parameters)
    
    final_values_arr, final_values_error_arr, fitting_ratio_parameters, fitting_ratio_error = work_out_g_factor(kittel_parameters, kittel_error, fitting_ratio_parameters, fitting_ratio_error)
    
    results = val_dictionary(peak_vals_arr, lorentz_error_arr, freq_arr, final_values_arr, final_values_error_arr, peak_width_arr, 
                             peak_width_error_arr, fitting_ratio_parameters, fitting_ratio_error)
    
    results={"20GHz_peak": results["20GHz_peak"], #this would be the peak position at 20GHz
             "20GHz_peak_error": results["20GHz_peak_error"], # uncertainty in above
             "20GHz_width": results["20GHz_width"], #Delta H for 20 GHz
             "20GHz_width_error": results["20GHz_width_error"],  # uncertainty in above
             "gamma": results["gamma"], #your gamma value
             "gamma_error": results["gamma_error"],  # uncertainty in above
             "g": results["g"], #Your Lande g factor
             "g_error": results["g_error"],  # uncertainty in above
             "Hk": results["Hk"], #Your value for the anisotropy field
             "Hk_error": results["Hk_error"],  # uncertainty in above
             "Ms": results["Ms"], #Your value for the saturation Magnetisation
             "Ms_error": results["Ms_error"],  # uncertainty in above
             "DeltaH": results["DeltaH"], #Your value for the intrinsic line width
             "DeltaH_error": results["DeltaH_error"],  # uncertainty in above
             "alpha": results["alpha"], # Your value for the Gilbert Damping parameter
             "alpha_error": results["alpha_error"] } # uncertainty in above

    #If your code doesn't find all of these values, just leave them set to None
    #Otherwise return the number. Your code does not have to round these numbers corr4ectly, but you must quote them
    #rounded correctly in your report.
    return results

if __name__=="__main__":
    # Put your test code in side this if statement to stop it being run when you import your code
    #Please avoid using input as the testing is going to be done by a computer programme, so
    #can't input things from a keyboard....
    filename='assessment_data_py16jo.dat'
    
    try:
        test_results=ProcessData(filename)
        
    except IOError:
        raise IOError('File could not be found')
        
#8e7aa605-094f-406c-844b-4367815591d3
    
