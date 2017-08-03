import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;

import javax.imageio.ImageIO;

import java.io.File;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;

public class Main{
	
	/*IMPORTANT MESSAGE: 

		You need my version of  bag1steering.txt to run this script!
		
		Nanoseconds have this weird indenting issue where they some times have 2 or 3 spaces 
		instead of 1 space between it's name "nsecs:" and its value.
		
		If you want to this for other files you will have to find and replace all the extra spaces with 0s
		
		Example (" " = @)
		so 		  "nsecs:@@@67322"
		would be: "nsecs:@0067322"
		
	*/
	
	// File representing the folder that you select using a FileChooser
	static final File dir = new File("/home/blackjack/Documents/center/"); //CHANGE THIS TO YOUR DIRECTORY LOCATION OF THE IMAGES
	
	
	//To modify this program for a new text file, modify these variables
	public static int TotalImages =11050; // total number of images in a given text document (roughly 1/19th number of lines or ~1/18.982)
	public static int TotalLines =209760; //total lines in text document


	// array of supported extensions (use a List if you prefer)
	static final String[] EXTENSIONS = new String[]{
	    "jpg" // and other formats you need
	};
	// filter to identify images based on their extensions
	static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

	    @Override
	    public boolean accept(final File dir, final String name) {
	        for (final String ext : EXTENSIONS) {
	        	//Search for file extensions that end with any of the extensions listed in EXTENSIONS
	            if (name.endsWith("." + ext)) {
	                return (true);
	            }
	        }
	        return (false);
	    }
	};
	
    public static void main(String[] args) throws IOException {
    	
    	//Setup FileReader and read in the Bag 1 steering file
    	File file1 = new File("bag1steering.txt"); //Modify this string to chose which file to read in for processing
    	FileReader fileReader = new FileReader(file1);
    	BufferedReader reader1 = new BufferedReader(fileReader);
    
    	//Create an array to store the completed file names + angles
    	String[] FileList = new String[TotalImages];
    	for(int k =0; k<TotalImages; k++) FileList[k]="";
    	int StringNumber = 0; 
    	
    	int[] SecList = new int[TotalImages];
    	int[] nSecList = new int[TotalImages];
    	String[] AngleList = new String[TotalImages];

    	System.out.println("Reading through Text document");
    	
    	//Run through all the lines in the Steering File
    	for(int i=0; i<TotalLines; i++) {
    	
    	//Read in current line and split it into a string of words
    	String temp1 = reader1.readLine();
    	String[] Split1 = temp1.split(" ");
    	
    	//This booleans determine whether or not a specific line has been found: E.g Secs, NanoSeconds,etc
    	Boolean Found_Secs  = false;
    	Boolean Found_nSecs  = false;
    	Boolean Found_Angle = false;
    	
    	//Goes through every word in the current line of the document-A.K.A Split1- and tries to find the critical information
	    	for(String n : Split1) {
	    			
	    			if(Found_Secs) {
						FileList[StringNumber] = FileList[StringNumber] + n+"."; //modify the "." if you have images with underscores instead of periods.
						SecList[StringNumber] = Integer.parseInt(n);
						Found_Secs = false;
					}
	    			if(Found_nSecs ) {
	    				//Splits up the current word into individual strings, and then creates a new string with the first 3 numbers with a for loop
	    				String[] newWord = n.split("");
	    				String ThreeDecimals = "";
	    				for(int j=0; j<3; j++)ThreeDecimals = ThreeDecimals + newWord[j];
	    				
	    				int Nano = Integer.parseInt(ThreeDecimals);
						nSecList[StringNumber] = Nano;
	    				
	    				//Adds the 3 digit number to the current String and a ".jpg " at the end
						FileList[StringNumber] = FileList[StringNumber] + ThreeDecimals+".jpg "; //Modify the .jpg if you have .pngs or any of the other vastly superior image formats.
						Found_nSecs = false;
					}
	    			if(Found_Angle) {
						FileList[StringNumber] = FileList[StringNumber] + n;
						AngleList[StringNumber] = n;
						//System.out.println(FileList[StringNumber]); //Comment this out if you want to increase performance and not see what is being directly written to the "Sorted Data File"
						StringNumber++; //increments when the file name and angle are added to current String object in the FileList Array
						Found_Angle = false;
					}
	    				//If it finds one of the words we are looking for/need, (secs:, nsecs:, etc) then we set their respective booleans to be true so we can add the next words (the numbers) we need.
						if(n.equals("secs:"))Found_Secs =  true;
						if(n.equals("nsecs:"))Found_nSecs =true;
						if(n.equals("steering_wheel_angle:")) Found_Angle = true;
	    	}
	    }
    	
    	//Create arrays to store all the Secs: and nSecs: values for the pictures
		int[] SecList2 = new int[TotalImages];
    	int[] nSecList2 = new int[TotalImages];
    	int entry =0;
		
    	System.out.println("Starting Image Cataloging");
		if (dir.isDirectory()) { // make sure it's a directory
		    for (final File f : dir.listFiles(IMAGE_FILTER)) {
		        BufferedImage img = null;

		        try {
		        	
		            img = ImageIO.read(f); //Reads in an image

		           String rawName = f.getName();//Gets the name of the image
		           String[] SplitParse = rawName.split(""); //Splits into a string of individual strings
		           String Seconds = "";
		           String nSeconds = "";
			          for(int l =0; l<10; l++) Seconds = Seconds + SplitParse[l]; //Takes the first 10 strings and creates the Seconds
			          for(int l =11; l<14; l++) nSeconds = nSeconds + SplitParse[l]; //Takes strings 12-14 and creates the nanoSeconds
			          int tempSeconds = Integer.parseInt(Seconds);
			          int tempNSeconds = Integer.parseInt(nSeconds);
		          
			      //Saves the Values found    
		          SecList2[entry] = tempSeconds;
		          nSecList2[entry] = tempNSeconds;
		          entry++;
		        	
		        } catch (final IOException e) {
		        	}
		    }
		    
	//This will keep track where everything is in the SecList array	 (Array of Seconds in the Bag1Steering Document)
    int currentIndex = 11;
    System.out.println("Starting matching");
		 
	//Goes through all the different values for Seconds in the Folder
	for( int currentSecond = 1479424215; currentSecond < 1479424436;currentSecond++) {	    
		
		int tempIndex = currentIndex;
		//sees how many pictures have the same number of seconds as the one time currently being examined 
		for(int i =0; i<60; i++) {
		    	if(SecList[(currentIndex+i)] != currentSecond) {
		    		currentIndex = currentIndex +i;
		    		break;
		    	}
		    }
		  
		//Looks through all nsecs values in the Bag1Steering for the current second being exmained and see if it's in a 10 nanosecond range of one of the images. 
		   for(int k =tempIndex; k<currentIndex; k++) {
		    for(int i =0; i<TotalImages; i++) {
		    	if(SecList2[i] ==currentSecond) {
		    		if(nSecList2[i] - nSecList[k]<=10 && nSecList2[i] - nSecList[k]>=-10 ) { //You can change the values of 10 and -10, but they work just fine honestly
		    			FileList[k] = SecList2[i] + "."+ nSecList2[i]+".jpg "+ AngleList[k];
		    			k++;
		    			i=0;
		    			break;
		    		}
		    		
		    	}
		     }
		   }
		 }
		}
		
		//Create a new file ("SortedData.txt" and filewriter 
    	File file2 = new File("SortedData.txt"); //If you'd like the output name change "Sorted Data"
		FileWriter filewriter = new FileWriter(file2);
		BufferedWriter writer= new BufferedWriter(filewriter);
			
		//Writes every single file and it's steering angle on the same line, then goes to a new line.
			for(int i =0; i<StringNumber+1; i++) {
				writer.write(FileList[i]);
				writer.newLine();
			}
		
		writer.flush();
		writer.close();
		
		System.out.println("Sort is finished");
		//The Program ends, that's all folks.
		    	
    }
}


