package data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ContinuousDiscretized extends SparseDataset{
	// Comma-separated
	private String delim = ",";
        private String delim2 = ";";
	String name;
        String level;

	
        //testing continuous data
        //mean split on wine
        public static class ContinuousWine extends ContinuousDiscretized{
                public ContinuousWine(){super("continuouswine" , "mean_split_binary_binning" , "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/winequality-white.csv");}
        }
        
        //mdlp split on wine
        public static class BigBinaryWine extends ContinuousDiscretized{
                public BigBinaryWine(){super("bigbinarywine", "mdlp_binary_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/winequality-white.csv");}
        }
        
        
        public static class EqualWidthBinWine extends ContinuousDiscretized{
                public EqualWidthBinWine(){super("equalwidthbinwine", "equal_width_bin_wine", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/winequality-white.csv ");}
        }
        
        public static class EwBinOne extends ContinuousDiscretized{
                public EwBinOne(){super("ewbinone", "equal_width_bin_wine_one", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/winequality-white.csv ");}
        }
        
        public static class EwCloud extends ContinuousDiscretized{
                public EwCloud(){super("ewcloud", "equal_width_binning","/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cloud-dataset.csv ");}
        }
        
        public static class EwStatlog extends ContinuousDiscretized{
                public EwStatlog(){super("ewstatlog", "equal_width_binning","/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/statlog-dataset.csv ");}
        }
        
        public static class EwBoone extends ContinuousDiscretized{
                public EwBoone(){super("ewboone", "equal_width_binning","/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/boone-dataset.csv ");}
        }
        
        public static class EwAnnealU extends ContinuousDiscretized{
                public EwAnnealU(){super("ewanneal", "equal_width_binning","/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/anneal-U-dataset.csv");}
        }
        
        public static class EwAustralian extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwAustralian(){super("ewaussie", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/australian-dataset.csv");}
                
        }
        
        public static class EwAuto extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwAuto(){super("ewauto", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/auto-dataset.csv");}
                
        }
        
        public static class EwCars extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwCars(){super("ewcars", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cars-dataset.csv");}
                
        }
        
        public static class EwCleave extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwCleave(){super("ewcleave", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cleave-dataset.csv");}
                
        }
        
        public static class EwCRX extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwCRX(){super("EwCRX", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/crx-dataset.csv");}
                
        }
        
        public static class EwDiabetes extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwDiabetes(){super("ewdiabetes", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/diabetes-dataset.csv");}
                
        }
        
        public static class EwGerman extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwGerman(){super("ewgerman", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/german-dataset.csv");}
                
        }
        
        public static class EwGermanOrg extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwGermanOrg(){super("ewgermanorg", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/germanorg-dataset.csv");}
                
        }
        
        public static class EwHeart extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwHeart(){super("ewheart", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/heat-dataset.csv");}
                
        }
        
        public static class EwIris extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwIris(){super("ewiris", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/iris-dataset.csv");}
                
        }
        
        public static class EwBurkina extends ContinuousDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public EwBurkina(){super("ewburkina", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/pcaburkina.csv");}
                
        }
        
        
        
        public static class EqualFrequencyBinWine extends ContinuousDiscretized{
                public EqualFrequencyBinWine(){super("equalfrequencybinwine", "easyef","");}
        }
        
        


	private ContinuousDiscretized(String name,  String level, String filepath) {
		this.name = name;
                this.level = level;
                

                int numDim = 5;
                int bins = 10;

                dims = new int[numDim];
                String dataset = filepath;
                
                
                //run python code to build datasets for continuous wine
                //check if dataset exits first though
                File file = new File(dataset);
                if(true){//
                    System.out.println("Building dataset");
                
                    //int[] onehot = new int[] {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}; //boone?
                    //int[] onehot = new int[]{1,1,1,1,1,1,1,1,1,0}; //statlog
                    //int[] onehot = new int[]{0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0}; //annealU
                    //int[] onehot =  new int[]{0,1,1,0,0,0,1,0,0,1,0,0,1,1,0}; //australian
                    //int[] onehot = new int[]{1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0}; //auto
                    //int[] onehot = new int[]{1,0,1,1,1,1,1,0,0}; //cars
                    //int[] onehot = new int[]{1,0,0,1,1,0,0,1,0,1,0,0,0,0}; //cleave
                    //int[] onehot = new int[]{0,1,1,0,0,0,0,1,0,0,1,0,0,1,1,0}; //crx
                    //int[] onehot = new int[]{0,1,1,1,1,1,1,1,0}; //diabetes
                    //int[] onehot = new int[]{0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}; //german
                    //int[] onehot = new int[]{0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; //germanorg
                    //int[] onehot = new int[]{1,0,0,1,1,0,0,1,0,1,0,0,0,0}; //heart
                    //int[] onehot = new int[]{1,1,1,1,0}; //iris
                    int[] onehot = new int[]{1,1,1,1,1,0}; //pcaburkina
                    
                    
                    String onehotString = Arrays.toString(onehot)
                        .replace(",", "")  //remove the commas
                        .replace("[", "")  //remove the right bracket
                        .replace("]", "")  //remove the left bracket
                        .trim();
                    
                    
                    String cmd = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/PolyDiscretizeData.py ";

                    
                    String dest = "/home/andreas/Documents/MScAPR/SLSPN/data/ ";
                    //String level = "easy ";
                    String param = cmd + this.name +" " + dataset + " " + this.level+" "+ dest  + " --lista "  + onehotString + " --listb " + bins  ;

                    String command = "python " + param;
                    System.out.println(command);
                    try{
                        Process p = Runtime.getRuntime().exec(command);

                        try {
                            p.waitFor(); // Wait for the process to finish.
                        } catch (InterruptedException ex) {
                            Logger.getLogger(ContinuousDiscretized.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        System.out.println("Script executed successfully");
                    }catch(IOException ex){
                        System.out.println (ex.toString());
                        System.out.println("Could not find file " + cmd);
                    }
               
                }
                
		int c = 0;
                int d = 0;
                int t = 0;
                int old = 0;
                int feat =0;

		//System.out.println("Dataset "+name+", "+numVar+" vars, "+numTrain+"tr, "+numValid+"va, "+numTest+"te");
		String prefix = "data/";
		String trainfilename = prefix + name + ".ts.data";
		String validfilename = prefix + name + ".valid.data";
		String testfilename  = prefix + name + ".test.data";
                String dimfilename = prefix + name + ".dim.data";
                
                String typefilename = prefix  + name + ".type.data";
                String oldindexfilename = prefix  + name + ".oldindex.data";
                String featindexfilename = prefix  + name + ".featindex.data";
		try {
                    
                    
                    
                        // Read dimensions
			System.out.println("Loading dimensions...");
			BufferedReader br = new BufferedReader(new FileReader(dimfilename));
			String line = null;
			while ((line = br.readLine()) != null) {
                                //for(int r=0; r<numDim; r++){
                                    dims[d] = Integer.parseInt(line);
                                    d++;
                                //}
			}
			br.close();
                    
                } catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
                
                this.numVar = dims[1];
		this.numTraining = dims[2];
		this.numValidation = dims[3];
		this.numTesting = dims[4];
                System.out.println("Dataset "+name+", "+this.numVar+" vars, "+this.numTraining+"tr, "+this.numValidation+"va, "+this.numTesting+"te");
                attrSizes = new int[this.numVar];
                data = new int[numTraining + numValidation + numTesting][this.numVar];
                typepoints = new int[this.numVar];
                oldIndecies = new int[this.numVar];
                featureIndecies = new int[this.numVar];
                
                for(int i=0;i<this.numVar;i++){ attrSizes[i]=2; }
                
                try{
                        
			// Read training
			System.out.println("Loading training...");
			BufferedReader br = new BufferedReader(new FileReader(trainfilename));
                        String line = null;
			while ((line = br.readLine()) != null) {
				String toks[] = line.split(delim);
				for(int f=0; f<this.numVar; f++){
					// Where the permutation is used
					data[c][f] = Integer.parseInt(toks[f]);
				}
				c++;
			}
			br.close();

			// Read validation
			System.out.println("Loading validation...");
			br = new BufferedReader(new FileReader(validfilename));
			while ((line = br.readLine()) != null) {
				String toks[] = line.split(delim);
				for(int f=0; f<this.numVar; f++){
					data[c][f] = Integer.parseInt(toks[f]);
				}
				c++;
			}
			br.close();

			// Read testing
			System.out.println("Loading testing...");
			br = new BufferedReader(new FileReader(testfilename));
			br.readLine(); // skip first line
			while ((line = br.readLine()) != null) {
				String toks[] = line.split(delim);
				for(int f=0; f<this.numVar; f++){
					data[c][f] = Integer.parseInt(toks[f]);
				}
				c++;
			}
			br.close();
                        
                         //Read split points
			System.out.println("Loading var type points...");
			br = new BufferedReader(new FileReader(typefilename));
			while ((line = br.readLine()) != null) {
				//String toks[] = line.split(delim);
				//for(int f=0; f<this.numVar; f++){
                                
				typepoints[t] = Integer.parseInt(line);
                                
				//}
				t++;
			}
			br.close();
                        
                        System.out.println("Loading old indecies...");
			br = new BufferedReader(new FileReader(oldindexfilename));
			while ((line = br.readLine()) != null) {
				//String toks[] = line.split(delim);
				//for(int f=0; f<this.numVar; f++){
                                
				oldIndecies[old] = Integer.parseInt(line);
                                
				//}
				old++;
			}
			br.close();
                        
                        System.out.println("Loading feature indecies...");
			br = new BufferedReader(new FileReader(featindexfilename));
			while ((line = br.readLine()) != null) {
				//String toks[] = line.split(delim);
				//for(int f=0; f<this.numVar; f++){
                                
				featureIndecies[feat] = Integer.parseInt(line);
                                
				//}
				feat++;
			}
			br.close();
                        
                       
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

                

                
		makesparse();
	}


	@Override
	public String toString() {
		return name;
	}
        
        



}
