#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>
#include <math.h>

#define	CEIL(a,b)		((a+b-1)/b)
#define SWAP(a,b,t)		t=b; b=a; a=t;
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))
#define PI 3.14159265
typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

uch *TheImg, *CopyImg;					// Where images are stored in CPU
uch *GPUImg, *GPUCopyImg, *GPUResult;	// Where images are stored in GPU

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



// Kernel that flips the given image horizontally
// each thread only flips a single pixel (R,G,B)
__global__
void imrotate(uch *ImgDst, uch *ImgSrc, ui Vpixels, ui Hpixels, ui BlkPerRow, ui RowBytes, double cosRot, double sinRot)
{	
	__shared__ uch PixBuffer[3072*16];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYsrcOffset = MYrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;

	////////////// find destination index	
	int c, h, v, X, Y, NewCol, NewRow;
	double newX, newY, H, V, Diagonal, ScaleFactor;

	c=MYcol;  		h=Hpixels/2;   v=Vpixels/2;	// integer div
	X=(double)c-(double)h;
	Y=(double)v-(double)MYrow;
	
	// pixel rotation matrix
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	
	// Scale to fit everything in the image box
	H=(double)Hpixels;
	V=(double)Vpixels;
	Diagonal=sqrt(H*H+V*V);
	ScaleFactor=(Hpixels>Vpixels) ? V/Diagonal : H/Diagonal;
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	
	// convert back from Cartesian to image coordinates
	NewCol=((int) newX+h);
	NewRow=v-(int)newY;
	ui MYdstOffset = NewRow*RowBytes;
	ui MYdstIndex = MYdstOffset + 3 * NewCol;
	///////////////	
	ui Mytid3 = MYtid*3;
	PixBuffer[Mytid3] = ImgSrc[MYsrcIndex];
	PixBuffer[Mytid3+1] = ImgSrc[MYsrcIndex+1];
	PixBuffer[Mytid3+2] = ImgSrc[MYsrcIndex+2];
	__syncthreads();

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = PixBuffer[Mytid3];
	ImgDst[MYdstIndex + 1] = PixBuffer[Mytid3+1];
	ImgDst[MYdstIndex + 2] = PixBuffer[Mytid3+2];
}






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
	printf("\n Input File name: %17s  (%d x %d)   File Size=%lu", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}


// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uch *Img, char* fn)
{
	FILE* f = fopen(fn, "wb");
	if (f == NULL){ printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	//write header
	fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
	//write data
	fwrite(Img, sizeof(uch), IMAGESIZE, f);
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%lu", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}


int main(int argc, char **argv)
{
	// char			Flip = 'H';
	float			tmpKernelExcutionTime, totalKernelExecutionTime; // GPU code run times
	cudaError_t		cudaStatus, cudaStatus2;
	cudaEvent_t		time1, time2;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow;
	// ui 			BlkPerRowInt, BlkPerRowInt2;
	ui				ThrPerBlk = 128, NumBlocks;
	// ui 				NB2, NB4, NB8, RowInts;
	ui				RowBytes;
	cudaDeviceProp	GPUprop;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	// ui				*GPUCopyImg32, *GPUImg32;
	char			SupportedBlocks[100];
	// int				KernelNum=1;
	char			KernelName[255];
	double			RotAngle, deltaAngle;					// rotation angle
	int 			RotIter;
	int 			TotalIters;
	double 			cosRot, sinRot;
	strcpy(ProgName, "imrotateG");
	if(argc!=4){
		printf("\n\nUsage: ./imrotateG infile outfile N");
		return 0;
	}
	strcpy(InputFileName, argv[1]);
	strcpy(OutputFileName, argv[2]);

	// Create CPU memory to store the input and output images
	TheImg = ReadBMPlin(InputFileName); // Read the input image if memory can be allocated
	if (TheImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	CopyImg = (uch *)malloc(IMAGESIZE);
	if (CopyImg == NULL){
		free(TheImg);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%lu %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	// Allocate GPU buffer for the input and output images	
	cudaStatus = cudaMalloc((void**)&GPUImg, IMAGESIZE);
	cudaStatus2 = cudaMalloc((void**)&GPUCopyImg, IMAGESIZE);
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}
	RowBytes = (IPH * 3 + 3) & (~3);
	RowBytes = (IPH * 3 + 3) & (~3);
	BlkPerRow = CEIL(IPH,ThrPerBlk);
	NumBlocks = IPV*BlkPerRow; 

	printf("\nNum blocks: %d\n", NumBlocks);
	printf("\nThread per block: %d\n", ThrPerBlk);
	TotalIters = atoi(argv[3]);
	if(TotalIters > 30){
		printf("\nN is too large, should be less or equal to 30\n");
	}
	deltaAngle = 2*PI/float(TotalIters);
	printf("\nTotal iterations: %d\n", TotalIters);

	// iteration to find all images

	strcpy(OutputFileName, argv[2]);
	char* token = strtok(OutputFileName, ".");
	char* OutputFirstName = token;
	token = strtok(NULL, ".");
	char* OutputLastName = token;

	for(RotIter=1; RotIter<=TotalIters; RotIter++){
		char outName[128]="";		
		char tmp[10];
		sprintf(tmp, "%d", RotIter);
		strcat(outName, OutputFirstName);
		strcat(outName, tmp);
		strcat(outName, ".");
		strcat(outName, OutputLastName);

		cudaEventCreate(&time1);
		cudaEventCreate(&time2);
		cudaEventRecord(time1, 0); // record time1 in the first iteration

		RotAngle = (double)(RotIter-1)*deltaAngle;
		cosRot = cos(RotAngle);
		sinRot = sin(RotAngle);
		printf("\nRotation angle = %lf\n", RotAngle);
		imrotate <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPV, IPH, BlkPerRow, RowBytes, cosRot, sinRot);

		cudaEventRecord(time2, 0); //record time2 in teh last iteration
		cudaEventSynchronize(time1);
		cudaEventSynchronize(time2);
		cudaEventElapsedTime(&tmpKernelExcutionTime, time1, time2);
		totalKernelExecutionTime += tmpKernelExcutionTime;



		strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}
		GPUResult = GPUCopyImg;
		cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
			exit(EXIT_FAILURE);
		}
		cudaStatus = cudaDeviceSynchronize();
			//checkError(cudaGetLastError());	// screen for errors in kernel launches
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
			free(TheImg);
			free(CopyImg);
			exit(EXIT_FAILURE);
		}
		WriteBMPlin(CopyImg, outName);		// Write the flipped image back to disk
		memset(CopyImg, 0, IMAGESIZE);
		cudaMemset(GPUCopyImg, 0, IMAGESIZE);

	}

	printf("\nTotal Kernel Execution    =%7.2f ms\n", totalKernelExecutionTime);
	
	cudaFree(GPUImg);
	cudaFree(GPUCopyImg);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	free(TheImg);
	free(CopyImg);
	return(EXIT_SUCCESS);
}



