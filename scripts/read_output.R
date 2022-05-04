#######################################
# R script to read outputs from RACCOON
# J. O. Nash      @2022
# F. Comitani     @2022
#######################################


open.h5<-function(file){ 

    #This function reads in an hdf5 file
    #
    # args: 
    #   file (hdf5), path to the hdf5 file to read
    # returns: 
    #   (hdf5 object), the hdf5 object
  
    library(rhdf5)

    print("Processing Data")
    
    name<-h5ls(file)$group[2]
    data<-h5read(file, name)
    h5closeAll()
    
    return(data)
  }

import_cluster_table<-function(cluster_data.h5){

  # This function imports the class assignment output from RACCOON, 
  # an hdf5 object, outputs an R dataframe. 
  #
  # args: 
  #   cluster_data (pandas dataframe hdf5, Python), path to the class assignment data hdf5 file
  # returns: 
  #   dataframe (R dataframe), columns are samples and cluster names are rows. 
  #                            Values are 1 when a sample is assigned to a cluster. 
  
  # Load the data
  cluster_data<-open.h5(cluster_data.h5)

  # Rename rows and columns
  names(cluster_data)[c(1:2, 4)]<-c("cluster_names", "samples", "cluster_values")
  rownames(cluster_data$cluster_values)<-cluster_data$cluster_names 
  colnames(cluster_data$cluster_values)<-cluster_data$samples
  
  # Keep only the dataframe
  cluster_data<-cluster_data$cluster_values

  return(cluster_data)
}

