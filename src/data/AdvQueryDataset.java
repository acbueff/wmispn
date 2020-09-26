/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

/**
 *
 * @author andreas
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class AdvQueryDataset extends Dataset {

	int q_data[][];
	int ev_data[][];
	int vals[];
        double adv_vals[];
        double adv_q_data[][];
        double adv_e_data[][];
	
	int numTraining, numValidation, numTesting;
	int numvars = 0;
        
        double pdfUnivariate[];
        int typepoints[];
        int oldIndecies[];
        int featIndecies[];
        
        //query related
        double query_vals[];
        ArrayList<Integer> queryIndex;
	
	public AdvQueryDataset(String queryfile, String evidencefile) {
		int numqueries = 1;
		String delim = ",";
		
		try {
			// Read training
			System.out.println("Loading query...");
			BufferedReader br = new BufferedReader(new FileReader(queryfile));
			String line = br.readLine();
			numvars = line.split(delim).length;
			while ((line = br.readLine()) != null) { numqueries++; }
			br.close();
			System.out.println("Found "+numqueries+" queries with "+numvars+" vars");
			numTraining = 0;
			numValidation = 0;
			numTesting = numqueries;
			
			q_data = new int[numqueries][numvars];
			ev_data = new int[numqueries][numvars];
                        query_vals = new double[numvars];
                        oldIndecies = new int[numvars];
                        typepoints = new int[numvars];
                        queryIndex = new ArrayList<>();
                        
                        String prefix = "data/";
                        String name ="datafilename";
                        
                        //String oldindexfilename = prefix  + "polycloud"+ ".oldindex.data";
                        //String typefilename = prefix  + "polycloud"+ ".type.data";
                        
                        //String oldindexfilename = prefix  + "polyboone"+ ".oldindex.data";
                        //String typefilename = prefix  + "polyboone"+ ".type.data";
                        
                        //String oldindexfilename = prefix  + "polystatlog"+ ".oldindex.data";
                       //String typefilename = prefix  + "polystatlog"+ ".type.data";
                        
                       //String oldindexfilename = prefix  + "betafake"+ ".oldindex.data";
                       //String typefilename = prefix  + "betafake"+ ".type.data";
                       
                       //String oldindexfilename = prefix  + "fakegauss"+ ".oldindex.data";
                       //String typefilename = prefix  + "fakegauss"+ ".type.data";
                       
                       //String oldindexfilename = prefix  + "fakeexpon"+ ".oldindex.data";
                       //String typefilename = prefix  + "fakeexpon"+ ".type.data";
                       
                       String oldindexfilename = prefix  + "fakemix"+ ".oldindex.data";
                       String typefilename = prefix  + "fakemix"+ ".type.data";
                       
                       
                        int old = 0;
                        int t = 0;
                        try{
                            
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
                        
                            
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (IOException e) {
                                e.printStackTrace();
                        }

			
			int c=0;
			br = new BufferedReader(new FileReader(queryfile));
			while ((line = br.readLine()) != null) {
				String toks[] = line.split(delim);
				for(int f=0; f<numvars; f++){
					if(toks[f].equals("*")){
						q_data[c][f] = -1;
					} else {
                                                System.out.println("toks["+f+"]" + toks[f]);
                                                //parse +/1 val for advanced query
                                                //adv_q_data[c][f] = +/- val
                                                //q_data[c][f] = 1 but pass values into datastructure
                                                double value = Double.parseDouble(toks[f]);
                                                System.out.println("value: "+ value);
                                                if(value != 1 && value != 0){
                                                    query_vals[f] = value; //collect doublevals and 1,0's'
                                                }
                                                if(value == 0){
                                                    q_data[c][f] = 0;
                                                }else{
                                                    q_data[c][f] = 1;
                                                }
                                                
                                                if(value != 1 && value !=0){
                                                    queryIndex.add(oldIndecies[f]);
                                                }else if(typepoints[f] == 0){//discrete case
                                                    queryIndex.add(oldIndecies[f]);
                                                }
					}
				}
				c++;
			}
			br.close();
			
			
			
			System.out.println("Loading evidence...");
			br = new BufferedReader(new FileReader(evidencefile));
			c=0;
			
			while ((line = br.readLine()) != null) {
				String toks[] = line.split(delim);
				for(int f=0; f<numvars; f++){
					if(toks[f].equals("*")){
						ev_data[c][f] = -1;
					} else {
                                                //parse +/- val for advanced query or maybe just a one hot encoded feature
                                                //adv_e_data[c][f] = +/- val
                                                //ev_data[c][f] = 1 but pass values into datastructure
                                                
						//ev_data[c][f] = Integer.parseInt(toks[f]);
                                                double value = Double.parseDouble(toks[f]);
                                                query_vals[f] = value; //collect doublevals and 1,0's'
                                                if(value == 0){
                                                    ev_data[c][f] = 0;
                                                }else{
                                                    ev_data[c][f] = 1;
                                                }
					}
				}
				c++;
			}
                        System.out.println("from evidence count" + c);
			br.close();
		} catch(Exception e){
			e.printStackTrace();
		}
		
		
	}
        
        public double[] getQueryValues(){
            return query_vals;//needs fixing
        }
        
        public ArrayList<Integer> getQueryIndexList(){
            return queryIndex;
        }
	
	//@Override
	public double getNumFolds() {
		// TODO Auto-generated method stub
		return 0;
	}

	//@Override
	public int getNumClasses() {
		// TODO Auto-generated method stub
		return 0;
	}

	//@Override
	public int getNumTesting() {
		return numTesting;
	}

	//@Override
	public int getNumTraining() {
		return numTraining;
	}

	//@Override
	public int getNumValidation() {
		return numValidation;
	}

	//@Override
	public int trueLabel() {
		return 0;
	}

	//@Override
	public void show(int i, Partition testing, boolean b) {
		// TODO Auto-generated method stub
	}

	public void showJoint(int i, Partition testing) {
		vals = q_data[i].clone();
		
		for(int j=0; j<numvars; j++){
                    //System.out.println("vals["+j+"]: " + vals[j]);
                   // System.out.println("ev_data["+i+"]["+j+"]: " + ev_data[i][j]);
			if(ev_data[i][j] > -1){
				vals[j] = ev_data[i][j]; 
                                //System.out.println("val["+j+"]: " + vals[j]);
			}
		}
	}
        
        
	
	public int getNumQueryVars(int i, Partition part){
		int c=0;
		for(int j=0; j<numvars; j++){
			if(q_data[i][j] > -1){
				c++;
			}
		}
		return c;
	}
	
	public void showQuery(int i, Partition testing) {
		vals = q_data[i];
	}
	
	public void showEvidencePlusAQuery(int inst, Partition testing, int var) {
		vals = ev_data[inst].clone();
		vals[var] = q_data[inst][var]; 
	}
        
        

	public void showEvidence(int i, Partition testing) {
		vals = ev_data[i];
	}
	
	//@Override
	public int getUniqueInstanceID() {
		// TODO Auto-generated method stub
		return 0;
	}

	//@Override
	public int getNumFeatures() {
		return numvars;
	}

	//@Override
	public int[] getAttrSizes() {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public int[] getValues() {
		return vals;
	}
        
        
        
        public double[] getPdfUnivariate(){
            return pdfUnivariate; //NOT instantiated
        }
        
        public int[] getVariableType(){
            return typepoints;
        }
        
        public int[] getOldIndecies(){
            return oldIndecies;
        }
        
        public int[] getFeatureIndecies(){
            return featIndecies;
        }
        
        public double[][] getMinMax(){
            
            return null;
        }
        public ArrayList<ArrayList<Double>> getPolyList(){
            
            return null;
        }

}