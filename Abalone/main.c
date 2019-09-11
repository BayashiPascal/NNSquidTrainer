#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "genalg.h"
#include "neuranet.h"
#include "thesquid.h"
#include "gdataset.h"

// Nb of step between each save of the GenAlg
// Saving it allows to restart a stop learning process but is 
// very time consuming if there are many input/hidden/output
// If 0 never save
#define SAVE_GA_EVERY 0
// Nb input and output of the NeuraNet
#define NB_INPUT 10
#define NB_OUTPUT 1
// Nb max of hidden values, links and base functions
#define NB_MAXHIDDEN 100  // 1
#define NB_MAXLINK 200  // 100
#define NB_MAXBASE NB_MAXLINK
// Size of the gene pool and elite pool
#define ADN_SIZE_POOL 500
#define ADN_SIZE_ELITE 20
// Initial best value during learning, must be lower than any
// possible value returned by Evaluate()
#define INIT_BEST_VAL -10000.0
// Value of the NeuraNet above which the learning process stops
#define STOP_LEARNING_AT_VAL -0.01
// Number of epoch above which the learning process stops
#define STOP_LEARNING_AT_EPOCH 1000
// Save NeuraNet in compact format
#define COMPACT true
// Time limit for one evaluation to complete on a Squidlet, in second
#define MAXWAIT 60
// Nb of Neuranet evaluated per task
#define BATCHSIZE 50

typedef enum DataSetCat {
  datalearn,
  datavalid,
  datatest,
  datasetCatSize
} DataSetCat;
const char* dataSetCatNames[datasetCatSize] = {
  "datalearn", "datavalid", "datatest"
  };

// Create an empty NeuraNet with the desired architecture
NeuraNet* CreateNN(void) {
  // Create the NeuraNet
  int nbIn = NB_INPUT;
  int nbOut = NB_OUTPUT;
  int nbMaxHid = NB_MAXHIDDEN;
  int nbMaxLink = NB_MAXLINK;
  int nbMaxBase = NB_MAXBASE;
  NeuraNet* nn = 
    NeuraNetCreate(nbIn, nbOut, nbMaxHid, nbMaxBase, nbMaxLink);

  // Return the NeuraNet
  return nn;
}

// CSVImporter for the dataset
void CSVImporter(
        int col, 
      char* val,
  VecFloat* sample) {

  if (col == 0) {

    if (*val == 'M') {

      VecSet(sample, 0, 1.0);
      VecSet(sample, 1, 0.0);
      VecSet(sample, 2, 0.0);

    } else if (*val == 'F') {

      VecSet(sample, 0, 0.0);
      VecSet(sample, 1, 1.0);
      VecSet(sample, 2, 0.0);

    } else if (*val == 'I') {

      VecSet(sample, 0, 0.0);
      VecSet(sample, 1, 0.0);
      VecSet(sample, 2, 1.0);

    }

  } else if (col == 8){

    VecSet(sample, 10, atof(val) + 0.5);

  } else {

    VecSet(sample, col + 2, atof(val));

  }
}

// Convert the original data at 'pathOrig' to a GDataSet and save it
// at 'pathRes'
// The GDataSet is splitted into 3 categories which dimensions are given
// by 'catSize'
// Return true if successful, else false
bool ConvertData(
  const char* const pathOrig,
  const char* const pathRes,
  const VecShort* const catSize) {

  // Declare a flag to memorize if the convertion was successful
  bool flag = false;

  // Declare the importer to convert the original CSV data
  unsigned int sizeHeader = 0;
  char sep = ' ';
  unsigned int nbCol = 9;
  unsigned int sizeSample = (NB_INPUT + NB_OUTPUT);
  GDSVecFloatCSVImporter importer = 
    GDSVecFloatCSVImporterCreateStatic(
      sizeHeader,
      sep,
      nbCol,
      sizeSample,
      &CSVImporter);

  // Import the data into a GDataSet
  GDataSetVecFloat dataset = 
    GDataSetCreateStaticFromCSV(
      pathOrig, 
      &importer);
  
  // If we could import the data
  int nbSample = GDSGetSize(&dataset);
  if (nbSample > 0) {

    // Split the samples into 3 categories of requested size
    // HAS NO EFFECT ON THE SAVED DATA
    GDSSplit(&dataset, catSize);

    // Open the result file
    FILE* fpRes = fopen(
      pathRes, 
      "w");
    
    // If we could open the result file
    if (fpRes != NULL) {
      
      // Save the converted dataset
      bool compact = true;
      flag = GDSSave(
        &dataset, 
        fpRes, 
        compact);
      
      // Close the result file
      fclose(fpRes);
    }

  }

  // Free memory
  GDataSetVecFloatFreeStatic(&dataset);
  
  // Return the success flag
  return flag;
}

// Learn 
void Learn(
  const char* const pathSquidletConf,
  const char* const pathData,
  const char* const pathWorkingDir,
  unsigned long int limitEpoch,
  float limitVal) {

  // Declare variables to measure time
  struct timespec start, stop;

  // Start measuring time
  clock_gettime(CLOCK_REALTIME, &start);

  // Create the NeuraNet
  NeuraNet* nn = CreateNN();

  // Declare a variable to memorize the best value
  float bestVal = INIT_BEST_VAL;

  // Create the Squad
  Squad* squad = SquadCreate();
  if (squad == NULL) {
    printf("Failed to create the squad\n");
    return;
  } else {
    FILE* fp = fopen(pathSquidletConf, "r");
    bool ret = SquadLoadSquidlets(squad, fp);
    if (ret == false) {
      printf("Failed to load the Squidlet config file\n");
      return;
    }
    fclose(fp);
  }
  //SquadSetFlagTextOMeter(squad, true);

  // Create the GenAlg used for learning
  // If previous weights are available in "./bestga.txt" reload them
  GenAlg* ga = NULL;
  FILE* fd = fopen("./bestga.txt", "r");
  if (fd) {

    printf("Reloading previous GenAlg...\n");
    if (!GALoad(&ga, fd)) {

      printf("Failed to reload the GenAlg.\n");
      NeuraNetFree(&nn);
      return;

    } else {

      printf("Previous GenAlg reloaded.\n");
      if (GABestAdnF(ga) != NULL)
        NNSetBases(nn, GABestAdnF(ga));
      if (GABestAdnI(ga) != NULL)
        NNSetLinks(nn, GABestAdnI(ga));

      GDataSetVecFloat dataset = GDataSetVecFloatCreateStaticFromFile(
        pathData);
      VecShort* inputs = VecShortCreate(NNGetNbInput(nn));
      for (unsigned int i = VecGetDim(inputs); i--;)
        VecSet(inputs, i, i);
      VecShort* outputs = VecShortCreate(NNGetNbOutput(nn));
      for (unsigned int i = VecGetDim(outputs); i--;)
        VecSet(outputs, i, VecGetDim(inputs) + i);
      unsigned long cat = 0;
      bestVal = GDSEvaluateNN(
        &dataset, 
        nn,
        cat,
        inputs,
        outputs,
        bestVal);
      VecFree(&inputs);
      VecFree(&outputs);
      GDataSetVecFloatFreeStatic(&dataset);

      printf("Starting with best at %f.\n", bestVal);
      limitEpoch += GAGetCurEpoch(ga);

    }
    fclose(fd);

  } else {

    printf("Creating new GenAlg...\n");
    fflush(stdout);
    ga = GenAlgCreate(ADN_SIZE_POOL, ADN_SIZE_ELITE, 
      NNGetGAAdnFloatLength(nn), NNGetGAAdnIntLength(nn));
    NNSetGABoundsBases(nn, ga);
    NNSetGABoundsLinks(nn, ga);
    // Must be declared as a GenAlg applied to a NeuraNet
    GASetTypeNeuraNet(ga, NB_INPUT, NB_MAXHIDDEN, NB_OUTPUT);
    GAInit(ga);

  }

  // If there is a NeuraNet available, reload it into the GenAlg
  fd = fopen("./bestnn.txt", "r");
  if (fd) {

    printf("Reloading previous NeuraNet...\n");
    if (!NNLoad(&nn, fd)) {

      printf("Failed to reload the NeuraNet.\n");
      NeuraNetFree(&nn);
      return;

    } else {

      printf("Previous NeuraNet reloaded.\n");

      GDataSetVecFloat dataset = GDataSetVecFloatCreateStaticFromFile(
        pathData);
      VecShort* inputs = VecShortCreate(NNGetNbInput(nn));
      for (unsigned int i = VecGetDim(inputs); i--;)
        VecSet(inputs, i, i);
      VecShort* outputs = VecShortCreate(NNGetNbOutput(nn));
      for (unsigned int i = VecGetDim(outputs); i--;)
        VecSet(outputs, i, VecGetDim(inputs) + i);
      unsigned long cat = 0;
      bestVal = GDSEvaluateNN(
        &dataset, 
        nn,
        cat,
        inputs,
        outputs,
        bestVal);
      VecFree(&inputs);
      VecFree(&outputs);
      GDataSetVecFloatFreeStatic(&dataset);

      printf("Starting with best at %f.\n", bestVal);
      GenAlgAdn* adn = GAAdn(ga, 0);
      if (adn->_adnF)
        VecCopy(adn->_adnF, nn->_bases);
      if (adn->_adnI)
        VecCopy(adn->_adnI, nn->_links);

    }
    fclose(fd);

  }

  // Start learning process
  printf("Learning...\n");
  printf("Will stop when curEpoch >= %lu or bestVal >= %f\n",
    limitEpoch, limitVal);
  printf("Will save the best NeuraNet in ./bestnn.txt at each improvement\n");
  fflush(stdout);

  // Declare a variable to memorize the best value in the current epoch
  float curBest = 0.0;
  float curWorst = 0.0;

  // Declare a variable to manage the save of GenAlg
  int delaySave = 0;

  // Learning loop
  while (bestVal < limitVal && 
    GAGetCurEpoch(ga) < limitEpoch) {
    curWorst = curBest;
    curBest = INIT_BEST_VAL;
    int curBestI = 0;
    unsigned long int ageBest = 0;

    // For each adn in the GenAlg
    for (int iEnt = 0; iEnt < GAGetNbAdns(ga); ++iEnt) {

      // Get the adn
      GenAlgAdn* adn = GAAdn(ga, iEnt);

      // Set the links and base functions of the NeuraNet according
      // to this adn
      if (GABestAdnF(ga) != NULL)
        NNSetBases(nn, GAAdnAdnF(adn));
      if (GABestAdnI(ga) != NULL)
        NNSetLinks(nn, GAAdnAdnI(adn));

      // Save the NeuraNet
      char nnFilename[100];
      sprintf(nnFilename, "nn%d.json", iEnt);
      char* pathNN = PBFSJoinPath(pathWorkingDir, nnFilename);
      FILE* fp = fopen(pathNN, "w");
      NNSave(nn, fp, true);
      fclose(fp);
      free(pathNN);
    }
      
    // For each adn in the GenAlg
    int cat = 0;
    for (int iEnt = 0; iEnt < GAGetNbAdns(ga); iEnt += BATCHSIZE) {
      
      VecLong* ids = VecLongCreate(
        MIN(GAGetNbAdns(ga), iEnt + BATCHSIZE) - iEnt);
      for (int jEnt = iEnt; 
        jEnt < MIN(GAGetNbAdns(ga), iEnt + BATCHSIZE);
        ++jEnt) {
        VecSet(ids, jEnt - iEnt, jEnt);
      }
      // Add the task to the Squad
      GenAlgAdn* adn = GAAdn(ga, iEnt);
      SquadAddTask_EvalNeuraNet(
        squad, 
        GAAdnGetId(adn), 
        MAXWAIT, 
        pathData, 
        pathWorkingDir, 
        ids, 
        curBest, 
        cat);
      VecFree(&ids);

    }

    // Loop until all the NeuraNet have been evaluated
    do {
      
      GSetSquadRunningTask completedTasks = SquadStep(squad);
      while (GSetNbElem(&completedTasks) > 0L) {

        SquadRunningTask* completedTask = GSetPop(&completedTasks);
        SquidletTaskRequest* task = completedTask->_request;
        if (strstr(task->_bufferResult, "\"success\":\"1\"") == NULL) {

          printf("squad : ");
          SquidletTaskRequestPrint(task, stdout);
          printf(" failed !!\n");
          return;

        } else {

          JSONNode* json = JSONCreate();
          JSONLoadFromStr(json, task->_bufferResult);
          JSONNode* propIds = JSONProperty(json, "nnids");
          JSONNode* propVal = JSONProperty(json, "v");
          VecLong* nnids = NULL;
          VecDecodeAsJSON(&nnids, propIds);
          VecFloat* vals = NULL;
          VecDecodeAsJSON(&vals, propVal);

          for (int iEnt = 0; iEnt < VecGetDim(nnids); ++iEnt) {
            unsigned long nnid = VecGet(nnids, iEnt);
            float value = VecGet(vals, iEnt);
            GenAlgAdn* adn = GAAdn(ga, nnid);
            GASetAdnValue(ga, adn, value);
            // Update the best value in the current epoch
            if (value > curBest) {
              curBest = value;
              curBestI = nnid;
              ageBest = GAAdnGetAge(adn);
            }
            if (value < curWorst)
              curWorst = value;

            char nnFilename[100];
            sprintf(nnFilename, "nn%ld.json", nnid);
            char* pathNN = PBFSJoinPath(pathWorkingDir, nnFilename);
            char cmd[1000];
            sprintf(cmd, "rm -f %s", pathNN);
            int retCmd = system(cmd);
            (void)retCmd;
            free(pathNN);
          }
          JSONFree(&json);
          VecFree(&nnids);
          VecFree(&vals);


        }

        SquadRunningTaskFree(&completedTask);

      }
      
    } while (SquadGetNbTaskToComplete(squad) > 0L);

    // Measure time
    clock_gettime(CLOCK_REALTIME, &stop);
    float elapsed = stop.tv_sec - start.tv_sec;
    int day = (int)floor(elapsed / 86400);
    elapsed -= (float)(day * 86400);
    int hour = (int)floor(elapsed / 3600);
    elapsed -= (float)(hour * 3600);
    int min = (int)floor(elapsed / 60);
    elapsed -= (float)(min * 60);
    int sec = (int)floor(elapsed);

    // If there has been improvement during this epoch
    if (curBest > bestVal) {

      bestVal = curBest;

      // Display info about the improvment
      printf("Improvement at epoch %05lu: %f(%03d) (in %02d:%02d:%02d:%02ds)       \n", 
        GAGetCurEpoch(ga), bestVal, curBestI, day, hour, min, sec);
      fflush(stdout);

      // Set the links and base functions of the NeuraNet according
      // to the best adn
      GenAlgAdn* bestAdn = GAAdn(ga, curBestI);
      if (GAAdnAdnF(bestAdn) != NULL)
        NNSetBases(nn, GAAdnAdnF(bestAdn));
      if (GAAdnAdnI(bestAdn) != NULL)
        NNSetLinks(nn, GAAdnAdnI(bestAdn));

      // Save the best NeuraNet
      fd = fopen("./bestnn.txt", "w");
      if (!NNSave(nn, fd, COMPACT)) {
        printf("Couldn't save the NeuraNet\n");
        NeuraNetFree(&nn);
        GenAlgFree(&ga);
        return;
      }
      fclose(fd);

    } else {

      fprintf(stderr, 
        "Epoch %05lu: v%f a%03lu(%03d) kt%03lu ", 
        GAGetCurEpoch(ga), curBest, ageBest, curBestI, 
        GAGetNbKTEvent(ga));
      fprintf(stderr, "(in %02d:%02d:%02d:%02ds)  \r", 
        day, hour, min, sec);
      fflush(stderr);

    }

    ++delaySave;

    if (SAVE_GA_EVERY != 0 && delaySave >= SAVE_GA_EVERY) {

      delaySave = 0;
      // Save the adns of the GenAlg, use a temporary file to avoid
      // loosing the previous one if something goes wrong during
      // writing, then replace the previous file with the temporary one
      fd = fopen("./bestga.tmp", "w");
      if (!GASave(ga, fd, COMPACT)) {
        printf("Couldn't save the GenAlg\n");
        NeuraNetFree(&nn);
        GenAlgFree(&ga);
        return;
      }
      fclose(fd);
      int ret = system("mv ./bestga.tmp ./bestga.txt");
      (void)ret;

    }

    // Step the GenAlg
    GAStep(ga);

  }

  // Free memory
  NeuraNetFree(&nn);
  GenAlgFree(&ga);
  SquadFree(&squad);

  // Measure time
  clock_gettime(CLOCK_REALTIME, &stop);
  float elapsed = stop.tv_sec - start.tv_sec;
  int day = (int)floor(elapsed / 86400);
  elapsed -= (float)(day * 86400);
  int hour = (int)floor(elapsed / 3600);
  elapsed -= (float)(hour * 3600);
  int min = (int)floor(elapsed / 60);
  elapsed -= (float)(min * 60);
  int sec = (int)floor(elapsed);
  printf("\nLearning complete (in %d:%d:%d:%ds)\n", 
    day, hour, min, sec);
  fflush(stdout);
}


int main(
  int argc, 
  char** argv) {

  // Declare a varibale to memorize the path to the squidlets 
  // configuration file
  char* squidletConfPath = NULL;

  // Declare a variable to memorize the limit in term of epoch
  unsigned long int limitEpoch = STOP_LEARNING_AT_EPOCH;

  // Declare a variable to memorize the limit in term of value
  float limitVal = STOP_LEARNING_AT_VAL;

  // Decode the prior arguments
  for (int iArg = 0; iArg < argc; ++iArg) {
    
    if (strcmp(argv[iArg], "-help") == 0) {

      printf("[-help] : display the help\n");
      printf("[-convData <pathOrig> <pathRes> ");
      printf("<nb samples learning> <nb samples valid> ");
      printf("<nb samples test>] : convert the data in the ");
      printf("file at <pathOrig> and save the resulting GDataSet ");
      printf("to 'pathRes'. The GDataSet is split into 3 categories ");
      printf("of dimensions <nb samples learning> <nb samples valid> ");
      printf("<nb samples test>]\n");
      printf("[-squidlets <path>] : path to the squidlets ");
      printf("configuration file\n");
      printf("-learn <path dataset> <path working dir> : learn from ");
      printf("the dataset located at <path dataset> and use the folder ");
      printf("<path working dir> for temporary files during learning");
      printf("[-epoch <limit epoch>] : max number of epoch during ");
      printf("learning");
      printf("[-val <limit value>] : max value during learning");
      return 0;

    } else if (strcmp(argv[iArg], "-squidlets") == 0 && 
      iArg < argc - 1) {
        
        squidletConfPath = argv[iArg + 1];

    } else if (strcmp(argv[iArg], "-epoch") == 0 && 
      iArg < argc - 1) {
        
        limitEpoch = atoi(argv[iArg + 1]);

    } else if (strcmp(argv[iArg], "-val") == 0 && 
      iArg < argc - 1) {
        
        limitVal = atof(argv[iArg + 1]);

    }
    
  }
  
  // Init the random generator
  srandom(time(NULL));

  // Decode the posterior arguments
  for (int iArg = 0; iArg < argc; ++iArg) {
    
    if (strcmp(argv[iArg], "-convData") == 0 && 
      iArg < argc - 5) {

      char* pathOrig = argv[iArg + 1];
      char* pathRes = argv[iArg + 2];
      VecShort* catSize = VecShortCreate(datasetCatSize);
      for (int iCatSize = datasetCatSize; iCatSize--;) {

        short size = atoi(
          argv[iArg + 3 + iCatSize]);
        if (size >= 0) {

          VecSet(
            catSize, 
            iCatSize,
            size);

        } else {

          printf("Invalid size of category: %d\n", size);
          return 1;

        }
      }
      bool flag = ConvertData(pathOrig, pathRes, catSize);
      VecFree(&catSize);
      if (flag == true) {

        printf("Successfully converted %s to %s\n", pathOrig, pathRes);

      } else {

        printf("Failed to convert %s to %s\n", pathOrig, pathRes);

      }

    } else if (strcmp(argv[iArg], "-learn") == 0 && 
      iArg < argc - 2) {
      
      char* pathLearnDataset = argv[iArg + 1];
      char* pathLearnWorkingDir = argv[iArg + 2];
      Learn(
        squidletConfPath,
        pathLearnDataset,
        pathLearnWorkingDir,
        limitEpoch,
        limitVal);

    }
    
  }

  // Return success code
  return 0;
}

