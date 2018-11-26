#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

ui origSize, TgtSize;
ui xshrink, yshrink;
ui origVpixels, origHpixels, origHbytes;

uch *TheImage, *NewImage;					// Where images are stored in CPU
uch *GPUSrcImage, *GPUTgtImage, *GPUResult;	// Where images are stored in GPU



struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;

#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)


// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin(char* fn)
{
	static uch *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL){	printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }

	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		ip.Hbytes = RowBytes;
	//save header for re-use
	memcpy(ip.HeaderInfo, HeaderInfo,54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}

void WriteBMPlin(uch *Img, char* fn)
{
	FILE* f = fopen(fn, "wb");
	if (f == NULL){ printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	//write header
	*(int*)&ip.HeaderInfo[2] = ip.Hbytes*ip.Vpixels+54;
	*(int*)&ip.HeaderInfo[18] = ip.Hpixels;
	*(int*)&ip.HeaderInfo[22] = ip.Vpixels;
	*(int*)&ip.HeaderInfo[34] = ip.Hbytes*ip.Vpixels;
	fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
	//write data
	fwrite(Img, sizeof(uch), IMAGESIZE, f);
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%u\n\n", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}
__global__
void ImShrunk(uch *TgtImg, uch *SrcImg, ui TgtHpixels, ui xshrink, ui yshrink, ui origHpixels, ui origHbytes)
{
	ui ThrPerBlk = blockDim.x;
	ui TgtBid = blockIdx.x;
	ui TgtTid = threadIdx.x;
	ui TgtGtid = ThrPerBlk * TgtBid + TgtTid;


	ui BlkPerRow = (TgtHpixels + ThrPerBlk - 1) / ThrPerBlk;  // ceil
	ui TgtRowBytes = (TgtHpixels * 3 + 3) & (~3);
	// ui SrcRowBytes = (origHpixels * 3 + 3) & (~3);
	ui SrcRowBytes = origHbytes;

	ui Tgtrow = TgtBid / BlkPerRow;
	ui TgtCol = TgtGtid - Tgtrow*BlkPerRow*ThrPerBlk;
	if(TgtCol >= TgtHpixels) return;
	
	ui SrcRow = Tgtrow * yshrink;
	ui SrcCol = TgtCol * xshrink;
	// if(SrcCol >= origHpixels) return;
	
	///////////////
	ui TgtOffset = Tgtrow * TgtRowBytes;
	ui SrcOffset = SrcRow * SrcRowBytes;

	ui TgtIndex = TgtOffset + 3*TgtCol;
	ui SrcIndex = SrcOffset + 3*SrcCol;

	TgtImg[TgtIndex] = SrcImg[SrcIndex];
	TgtImg[TgtIndex+1] = SrcImg[SrcIndex+1];
	TgtImg[TgtIndex+2] = SrcImg[SrcIndex+2];

}



int main(int argc, char** argv){
	
	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t cudaStatus, cudaStatus2;
	cudaEvent_t time1, time2, time3, time4;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks, GPUDataTransfer;
	cudaDeviceProp GPUprop;
	ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];

	strcpy(ProgName, "imShrunk");
	if(argc!=5){
		printf("\n\nUsage: imshrunk input output xshrink yshrink");
		return 0;
	}

	xshrink = atoi(argv[3]);
	yshrink = atoi(argv[4]);
	strcpy(InputFileName, argv[1]);
	strcpy(OutputFileName, argv[2]);



	TheImage = ReadBMPlin(argv[1]);
	if (TheImage == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	origHpixels = ip.Hpixels;
	origVpixels = ip.Vpixels;
	origHbytes = ip.Hbytes;
	origSize = origHbytes * origVpixels;

	ip.Hpixels = ip.Hpixels/xshrink;
	ip.Hbytes = (ip.Hpixels*3 + 3) & (~3);
	ip.Vpixels = ip.Vpixels/yshrink;
	// TgtSize = ip.Hbytes * ip.Vpixels;
	// printf("\n new Hpixels %u", ip.Hpixels);
	// printf("\n new Vpixels %u", ip.Vpixels);
	
	NewImage = (uch *)malloc(IMAGESIZE);
	if (NewImage == NULL){
		free(NewImage);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

		// Choose which GPU to run on, change this on a multi-GPU system.
		int NumGPUs = 0;
		cudaGetDeviceCount(&NumGPUs);
		if (NumGPUs == 0){
			printf("\nNo CUDA Device is available\n");
			exit(EXIT_FAILURE);
		}
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			exit(EXIT_FAILURE);
		}
		cudaGetDeviceProperties(&GPUprop, 0);
		SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
		SupportedMBlocks = SupportedKBlocks / 1024;
		sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
		MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;
	
		cudaEventCreate(&time1);
		cudaEventCreate(&time2);
		cudaEventCreate(&time3);
		cudaEventCreate(&time4);
	
		cudaEventRecord(time1, 0);

		// allocate GPU buffer
		cudaStatus = cudaMalloc((void**)&GPUSrcImage, origSize);
		cudaStatus2 = cudaMalloc((void**)&GPUTgtImage, IMAGESIZE);
		if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
			fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
			exit(EXIT_FAILURE);
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(GPUSrcImage, TheImage, origSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
			exit(EXIT_FAILURE);
		}
		
		cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
		BlkPerRow = (IPH + ThrPerBlk -1 ) / ThrPerBlk;
		NumBlocks = IPV*BlkPerRow;
	
		ImShrunk <<< NumBlocks, ThrPerBlk>>> (GPUTgtImage, GPUSrcImage, IPH, xshrink, yshrink, origHpixels, origHbytes);


		GPUResult = GPUTgtImage;
		GPUDataTransfer = origSize + IMAGESIZE;

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}
		cudaEventRecord(time3, 0);

			// Copy output (results) from GPU buffer to host (CPU) memory.
		cudaStatus = cudaMemcpy(NewImage, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
			exit(EXIT_FAILURE);
		}

		
		cudaEventRecord(time4, 0);

		cudaEventSynchronize(time1);
		cudaEventSynchronize(time2);
		cudaEventSynchronize(time3);
		cudaEventSynchronize(time4);

		cudaEventElapsedTime(&totalTime, time1, time4);
		cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
		cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
		cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

		cudaStatus = cudaDeviceSynchronize();

		//checkError(cudaGetLastError());	// screen for errors in kernel launches
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
			free(TheImage);
			free(NewImage);
			exit(EXIT_FAILURE);
		}
		WriteBMPlin(NewImage, argv[2]);		// Write the flipped image back to disk

		////////////////// change from here
		printf("\n\n--------------------------------------------------------------------------\n");
		printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n", 
				GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
		printf("--------------------------------------------------------------------------\n");
		printf("%s %s %s %u   [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName,
				ThrPerBlk, NumBlocks, BlkPerRow);
		printf("--------------------------------------------------------------------------\n");
		printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(origSize), DATABW(origSize, tfrCPUtoGPU));
		printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
		printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU));
		printf("--------------------------------------------------------------------------\n");
		printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((origSize + IMAGESIZE + GPUDataTransfer)), DATABW((origSize + IMAGESIZE+ GPUDataTransfer), totalTime));
		printf("--------------------------------------------------------------------------\n\n");

			// Deallocate CPU, GPU memory and destroy events.
		cudaFree(GPUSrcImage);
		cudaFree(GPUTgtImage);
		cudaEventDestroy(time1);
		cudaEventDestroy(time2);
		cudaEventDestroy(time3);
		cudaEventDestroy(time4);
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			free(TheImage);
			free(NewImage);
			exit(EXIT_FAILURE);
		}
		free(TheImage);
		free(NewImage);
		return(EXIT_SUCCESS);
}

