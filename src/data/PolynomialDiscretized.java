package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Arrays;

public class PolynomialDiscretized extends SparseDataset{
	// Comma-separated
	private String delim = ",";
        private String delim2 = ";";
        private String delim3 = " ";
	String name;
        String level;
        public static boolean dirty = false;

	
        //testing continuous  data with polynomials

        
        public static class PolyCloud extends PolynomialDiscretized{
                public PolyCloud(){super("polycloud", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cloud-dataset.csv",new int[]{1,1,1,1,1,1,1,1,1,1});}
                
        }
        
        public static class PolyStatlog extends PolynomialDiscretized{
                public PolyStatlog(){super("polystatlog", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/statlog-dataset.csv",new int[]{1,1,1,1,1,1,1,1,1,0});}
                
        }
        
        public static class PolyBoone extends PolynomialDiscretized{
                public PolyBoone(){super("polyboone", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/boone-dataset.csv",new int[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});}
                
        }
        
        public static class PolyAustralian extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyAustralian(){super("polyaussie", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/australian-dataset.csv",  new int[]{0,1,1,0,0,0,1,0,0,1,0,0,1,1,0});}
                
        }
        
        public static class PolyAnnealU extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyAnnealU(){super("polyanneal", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/anneal-U-dataset.csv",  new int[]{0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0});}
                
        }
        
        public static class PolyAuto extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyAuto(){super("polyauto", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/auto-dataset.csv",  new int[]{1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0});}
                
        }
        
        public static class PolyCars extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyCars(){super("polycars", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cars-dataset.csv",  new int[]{1,0,1,1,1,1,1,0,0});}
                
        }
        
        public static class PolyCleave extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyCleave(){super("polycleave", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/cleave-dataset.csv",  new int[]{1,0,0,1,1,0,0,1,0,1,0,0,0,0});}
                
        }
        
        public static class PolyCRX extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyCRX(){super("polyCRX", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/crx-dataset.csv",  new int[]{0,1,1,0,0,0,0,1,0,0,1,0,0,1,1,0});}
                
        }
        
        public static class PolyDiabetes extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyDiabetes(){super("polydiabetes", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/diabetes-dataset.csv",  new int[]{0,1,1,1,1,1,1,1,0});}
                
        }
        
        public static class PolyGerman extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyGerman(){super("polygerman", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/german-dataset.csv",  new int[]{0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0});}
                
        }
        
        public static class PolyGermanOrg extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyGermanOrg(){super("polygermanorg", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/germanorg-dataset.csv",  new int[]{0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});}
                
        }
        
        public static class PolyHeart extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyHeart(){super("polyheart", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/heat-dataset.csv",  new int[]{1,0,0,1,1,0,0,1,0,1,0,0,0,0});}
                
        }
        
        public static class PolyIris extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public PolyIris(){super("polyiris", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/iris-dataset.csv",  new int[]{1,1,1,1,0});}
                
        }

        public static class BetaFake extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public BetaFake(){super("betafake", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/fake.csv",  new int[]{1,1,1});}
                
        }
        
        public static class FakeGauss extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public FakeGauss(){super("fakegauss", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/fakegauss.csv",  new int[]{1});}
                
        }
        
        public static class FakeExpon extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public FakeExpon(){super("fakeExpon", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/fakeexpon.csv",  new int[]{1});}
                
        }
        
        public static class FakeMix extends PolynomialDiscretized{
                //1 = continuous feature for polynomial leaf, 0 = category feature ie 1hot encoded for monomial leaf
                public FakeMix(){super("fakemix", "equal_width_binning", "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/fakemix.csv",  new int[]{1});}
                
        }


	private PolynomialDiscretized(String name, String level, String filepath, int[] onehot) {
		this.name = name;
                this.level = level;
               //System.out.println(Arrays.toString(onehot));
                String onehotString = Arrays.toString(onehot)
                        .replace(",", "")  //remove the commas
                        .replace("[", "")  //remove the right bracket
                        .replace("]", "")  //remove the left bracket
                        .trim();
                System.out.println(onehotString);
                int numDim = 5;
                int bins = 2;
               

                dims = new int[numDim];
                String dataset = filepath;
                
                
                
                //run python code to build datasets for continuous wine
                File file = new File(dataset);
                if(true){
              
                    //String cmd = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/DiscretizeData.py ";
                    System.out.println("building dataset");
                    String cmd = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/PolyDiscretizeData.py ";

                    
                    String dest = "/home/andreas/Documents/MScAPR/SLSPN/data/ ";
                    //String level = "easy ";
                    String param = cmd + this.name +" " + dataset + " " + this.level+" "+ dest  + " --lista "  + onehotString + " --listb " + bins  ;

                    String command = "python " + param;
                    System.out.println(command);
                    try{
                        System.out.println("About to run the line");
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
                
                    
                    
                    
                  

		//for(int i=0;i<numVar;i++){ attrSizes[i]=2; }

		int c = 0;
                int d = 0;
                int t = 0;
                int old = 0;
                int feat = 0;

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
		this.numTesting = dims[3];
		this.numValidation = dims[4];
                System.out.println("Dataset "+name+", "+this.numVar+" vars, "+this.numTraining+"tr, "+this.numValidation+"va, "+this.numTesting+"te");
                attrSizes = new int[this.numVar];
                data = new int[numTraining + numValidation + numTesting][this.numVar];
                typepoints = new int[this.numVar];
                oldIndecies = new int[this.numVar];
                featureIndecies = new int[this.numVar];
                pdfUnivariate = new double[this.numVar];
                minMax = new double[2][this.numVar];
                polyList = new ArrayList<ArrayList<Double>>();
                int countPoly = 0;
                
                
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

                System.out.println("got this far");
                
                String pdffilename = prefix  + name + ".pdf.data";
                
                //int bval = 0;
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/cloud_ew_Better_intevalnr5/" + "cloud_ew_Better_intevalnr5.values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/cloud_ew_Better_intevalnr5/" + "cloud_ew_Better_intevalnr5.data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/cloud_ew_Better_intevalnr5/" + "cloud_ew_Better_intevalnr5.functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/statlog_ew_Better_intevalnr20/" + "statlog_ew_Better_intevalnr20.values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/statlog_ew_Better_intevalnr20/" + "statlog_ew_Better_intevalnr20.data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/statlog_ew_Better_intevalnr20/" + "statlog_ew_Better_intevalnr20.functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/boone_equalWidthNEW_intevalnr"+bins+"/" + "boone_equalWidthNEW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/boone_equalWidthNEW_intevalnr"+bins+"/" + "boone_equalWidthNEW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/boone_equalWidthNEW_intevalnr"+bins+"/" + "boone_equalWidthNEW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/annealUTrain_equalW_intevalnr"+bins+"/" + "annealUTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/annealUTrain_equalW_intevalnr"+bins+"/" + "annealUTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/annealUTrain_equalW_intevalnr"+bins+"/" + "annealUTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/australiaTrain_equalW_intevalnr"+bins+"/" + "australiaTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/australiaTrain_equalW_intevalnr"+bins+"/" + "australiaTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/australiaTrain_equalW_intevalnr"+bins+"/" + "australiaTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/autoTrain_equalW_intevalnr"+bins+"/" + "autoTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/autoTrain_equalW_intevalnr"+bins+"/" + "autoTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/autoTrain_equalW_intevalnr"+bins+"/" + "autoTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/carsTrain_equalW_intevalnr"+bins+"/" + "carsTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/carsTrain_equalW_intevalnr"+bins+"/" + "carsTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/carsTrain_equalW_intevalnr"+bins+"/" + "carsTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/cleveTrain_equalW_intevalnr"+bins+"/" + "cleveTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/cleveTrain_equalW_intevalnr"+bins+"/" + "cleveTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/cleveTrain_equalW_intevalnr"+bins+"/" + "cleveTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/crxTrain_equalW_intevalnr"+bins+"/" + "crxTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/crxTrain_equalW_intevalnr"+bins+"/" + "crxTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/crxTrain_equalW_intevalnr"+bins+"/" + "crxTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/diabetesTrain_equalW_intevalnr"+bins+"/" + "diabetesTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/diabetesTrain_equalW_intevalnr"+bins+"/" + "diabetesTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/diabetesTrain_equalW_intevalnr"+bins+"/" + "diabetesTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/germanTrain_equalW_intevalnr"+bins+"/" + "germanTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/germanTrain_equalW_intevalnr"+bins+"/" + "germanTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/germanTrain_equalW_intevalnr"+bins+"/" + "germanTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/germanorgTrain_equalW_intevalnr"+bins+"/" + "germanorgTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/germanorgTrain_equalW_intevalnr"+bins+"/" + "germanorgTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/germanorgTrain_equalW_intevalnr"+bins+"/" + "germanorgTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/heartTrain_equalW_intevalnr"+bins+"/" + "heartTrain_equalW_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/heartTrain_equalW_intevalnr"+bins+"/" + "heartTrain_equalW_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/heartTrain_equalW_intevalnr"+bins+"/" + "heartTrain_equalW_intevalnr"+bins+".functions";
                
                pdffilename  = "/home/andreas/Downloads/MSPNdatasets/irisTrain_equalW_intevalnr"+bins+"/" + "irisTrain_equalW_intevalnr"+bins+".values";
                String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/irisTrain_equalW_intevalnr"+bins+"/" + "irisTrain_equalW_intevalnr"+bins+".data";
                String polyfilename = "/home/andreas/Downloads/MSPNdatasets/irisTrain_equalW_intevalnr"+bins+"/" + "irisTrain_equalW_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/fake"+bins+"/" + "fake"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/fake"+bins+"/" + "fake"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/fake"+bins+"/" + "fake"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/gauss_intevalnr"+bins+"/" + "gauss_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/gauss_intevalnr"+bins+"/" + "gauss_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/gauss_intevalnr"+bins+"/" + "gauss_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/exp1000_intevalnr"+bins+"/" + "exp1000_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/exp1000_intevalnr"+bins+"/" + "exp1000_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/exp1000_intevalnr"+bins+"/" + "exp1000_intevalnr"+bins+".functions";
                
                //pdffilename  = "/home/andreas/Downloads/MSPNdatasets/beta-gauss_intevalnr"+bins+"/" + "beta-gauss_intevalnr"+bins+".values";
                //String binrangefilename = "/home/andreas/Downloads/MSPNdatasets/beta-gauss_intevalnr"+bins+"/" + "beta-gauss_intevalnr"+bins+".data";
                //String polyfilename = "/home/andreas/Downloads/MSPNdatasets/beta-gauss_intevalnr"+bins+"/" + "beta-gauss_intevalnr"+bins+".functions";
                
                //TEST
                int v = 0;
                    try {
                    
                        // Read pdfs
			System.out.println("Loading pdf values...");
			BufferedReader br = new BufferedReader(new FileReader(pdffilename));
			String line = null;
			//while ((line = br.readLine()) != null) {
                                //for(int r=0; r<numDim; r++){
                                    //pdfUnivariate[v1] = Integer.parseInt(line);
                                    //v1++;
                                //}
                                
                        for(int j = 0; j < this.numVar; j++){
                            if(typepoints[j] == 1){
                                line = br.readLine();
                                
                                if(line == null){
                                    System.out.println("empty line so break");
                                    break;
                                }
                                pdfUnivariate[j] = Double.parseDouble(line);
                                countPoly++;
                                //System.out.println("pdfUnivarate:" + pdfUnivariate[j]);
                            }
                            
                        }    
			br.close();
                        System.out.println(countPoly);
                        
                        int[] polyIndex = new int[countPoly];
                        for(int ones = 0; ones < this.numVar; ones++){
                            if(typepoints[ones] == 1){
                                polyIndex[v] = ones;
                                v++;
                            }
                        }
                        
                        
                        //Read min max for bins
                        System.out.println("Loading min max values...");
			br = new BufferedReader(new FileReader(binrangefilename));
                        int p1 = -1;
                        int p2 = -1;
                        double min = -1;
                        double max = -1;
                        for(int var = 0; var < this.numVar; var++){
                            if(typepoints[var] == 1){
     
                                line = br.readLine();
                                if(line == null){
                                    break;
                                }
                                p1 = 0;
                                p2 = 1;
                                String toks[] = line.split(delim3);                             
                                for(int b = 0; b < bins; b++){
                                    min = Double.parseDouble(toks[p1]);
                                    max = Double.parseDouble(toks[p2]);
                                    System.out.println("min: " + min + " max: " + max);
                                    p1++;
                                    p2++;
                                    minMax[0][var] = min;
                                    minMax[1][var] = max;
                                    if(b < (bins - 1)){
                                        var++;
                                    }
                                }                            
                                
                            }

                        }
                        
                                               
                        System.out.println("Loading polynomial values...");
			br = new BufferedReader(new FileReader(polyfilename));
                        boolean newbin = false;
                        //bins
                        //typepoints[i]
                        //countPoly
                        //polyIndex[j]
                        
                        //for(int p = 0; p < countPoly; p++){
                          //  line = br.readLine();
                            //if(line == null){
                              //      System.out.println("empty line so break");
                                //    break;
                             //   }
                            //if(!line.contains("#")){
                              //  String toks[] = line.split(delim3);
                               // ArrayList<Double> poly = new ArrayList<>();
                               // poly.add((double)polyIndex[p]);
                               // for(int r = 0; r < toks.length;r++){
                                //    poly.add(Double.parseDouble(toks[r]));
                               // }
                               // polyList.add(poly);
                            //}
                            
                            
                        line = br.readLine();
                        int perp = 0;
                        while(line != null){
                            if(!line.contains("#")){
                                String toks[] = line.split(delim3);
                                ArrayList<Double> poly = new ArrayList<>();
                                poly.add((double)polyIndex[perp]);
                                for(int r = 0; r < toks.length;r++){
                                    poly.add(Double.parseDouble(toks[r]));
                                }
                                polyList.add(poly);
                                perp++;
                            }
                            line = br.readLine();
                        }
                            
                            
                        
                        
                        
                        //print contents 
                     System.out.println("all polyList: " + Arrays.toString(polyIndex));   
                    for(ArrayList<Double> innerLs : polyList) {
                        for(double e : innerLs){
                            System.out.print(" " + e + " ");
                        }
                        System.out.println();
                    }
                    
                        
                    } catch (FileNotFoundException e) {
                            e.printStackTrace();
                    } catch (IOException e) {
                            e.printStackTrace();
                    }
                
                System.out.println("pdfs locked and loaded");
                
                System.out.println("all typepoints: " + Arrays.toString(typepoints));
                System.out.println("all oldIndecies: " + Arrays.toString(oldIndecies));
                System.out.println("all featIndecies: " + Arrays.toString(featureIndecies));
                System.out.println("all minMax: " + Arrays.deepToString(minMax));
                System.out.println("all pfdUnivariate: " + Arrays.toString(pdfUnivariate));
                //run R script, get univariate polynomial pdf's + poly functions
                file = new File(pdffilename);
                

                //file.exists()
                if(false){ //have not run polynomial code yet...check if file already exists
                    System.out.println("Beginning PMF capture for variables");
                    String param = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/IntegrationAndreas.py ";
                    String command = "python" + param;
                    System.out.println(command);
                    try{
                        Process p = Runtime.getRuntime().exec(command);

                        try {
                            p.waitFor(); // Wait for the process to finish.
                        } catch (InterruptedException ex) {
                            Logger.getLogger(ContinuousDiscretized.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        System.out.println("Script executed successfully");
                        
                        dirty = true;
                    }catch(IOException ex){
                        System.out.println (ex.toString());
                        System.out.println("Could not find file " + command);
                    }
                    
                    
                    
                    int v1 = 0;
                    int v2 = 0;
                    try {
                    
                        // Read dimensions
			System.out.println("Loading pdf values...");
			BufferedReader br = new BufferedReader(new FileReader(pdffilename));
			String line = null;
			//while ((line = br.readLine()) != null) {
                                //for(int r=0; r<numDim; r++){
                                    //pdfUnivariate[v1] = Integer.parseInt(line);
                                    //v1++;
                                //}
                                
                        for(int j = 0; j <= this.numVar; j++){
                            if(typepoints[j] == 1){
                                line = br.readLine();
                                
                                if(line == null){
                                    break;
                                }
                                pdfUnivariate[j] = Integer.parseInt(line);
                            }
                            
                        }        
                            
			//}
			br.close();
                        
                        
                        
                    
                    } catch (FileNotFoundException e) {
                            e.printStackTrace();
                    } catch (IOException e) {
                            e.printStackTrace();
                    }
                    
                }

                
		makesparse();
	}


	@Override
	public String toString() {
		return name;
	}



}

