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




// each thread only flips a single pixel (R,G,B)
// original version, 1D block
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

// each thread only flips a single pixel (R,G,B)
// use shared memory 3072 bytes, with less registers, put ScalerFactor outside the box, and 2D block
__global__
void imrotate2(uch *ImgDst, uch *ImgSrc, ui Vpixels, ui Hpixels, ui RowBytes, double cosRot, double sinRot, double ScaleFactor)
{	// use shared 
	__shared__ uch PixBuffer[3072]; // 1024 pixels

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	// ui MYgtid = ThrPerBlk * MYbid + MYtid;

	// ui MYrow = MYbid / BlkPerRow;
	// ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYsrcOffset = MYrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;

	////////////// find destination index	
	int X, Y, NewCol, NewRow;
	double newX, newY;
	// double Diagonal, ScaleFactor;

	// c=MYcol;  		
	// h=Hpixels/2;   
	// v=Vpixels/2;	// integer div
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	
	// pixel rotation matrix
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	
	// Scale to fit everything in the image box
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	
	// convert back from Cartesian to image coordinates
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
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

// each thread only flips a single pixel (R,G,B)
// use shared memory 6144 bytes, with less registers, put ScalerFactor outside the box, and 2D block
__global__
void imrotate3(uch *ImgDst, uch *ImgSrc, ui Vpixels, ui Hpixels, ui RowBytes, double cosRot, double sinRot, double ScaleFactor)
{	
	__shared__ uch PixBuffer[3072*2];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYsrcOffset = MYrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;

	////////////// find destination index	
	int X, Y, NewCol, NewRow;
	double newX, newY;

	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	
	// pixel rotation matrix
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	
	// Scale to fit everything in the image box
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	
	// convert back from Cartesian to image coordinates
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
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

// each thread only flips 2 single pixel (R,G,B)
// use shared memory 4096*4 pixels, with less registers, put ScalerFactor outside the box, and 2D block
__global__
void imrotate4(uch *ImgDst, uch *ImgSrc, ui Vpixels, ui Hpixels, ui RowBytes, double cosRot, double sinRot, double ScaleFactor)
{	
	__shared__ uch PixBuffer[3072*4];
	// if not enough shared memory, reduce the number of pix buffers
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYsrcOffset = MYrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	// ui iter = 0;

	////////////// find destination index
	int X, Y, NewCol, NewRow, MYcol2;
	double newX, newY;
	ui MYdstOffset2, MYdstIndex2;
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;	
	// pixel rotation matrix
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;	
	// Scale to fit everything in the image box
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;	
	// convert back from Cartesian to image coordinates
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
	ui MYdstOffset = NewRow*RowBytes;
	ui MYdstIndex = MYdstOffset + 3 * NewCol;
	///////////////	
	ui Mytid3 = MYtid*3;
	MYcol2=MYcol+1;


	PixBuffer[Mytid3] = ImgSrc[MYsrcIndex];
	PixBuffer[Mytid3+1] = ImgSrc[MYsrcIndex+1];
	PixBuffer[Mytid3+2] = ImgSrc[MYsrcIndex+2];

	if(MYcol2 < Hpixels){
		X=(double)MYcol-(double)(Hpixels/2);
		// Y=(double)(Vpixels/2)-(double)MYrow;	
		// pixel rotation matrix
		newX=cosRot*X-sinRot*Y;
		newY=sinRot*X+cosRot*Y;	
		// Scale to fit everything in the image box
		newX=newX*ScaleFactor;
		newY=newY*ScaleFactor;	
		// convert back from Cartesian to image coordinates
		NewCol=((int) (newX+Hpixels/2));
		NewRow=Vpixels/2-(int)newY;
		MYdstOffset2 = NewRow*RowBytes;
		MYdstIndex2 = MYdstOffset2 + 3 * NewCol;
		PixBuffer[Mytid3+3] = ImgSrc[MYsrcIndex+3];
		PixBuffer[Mytid3+4] = ImgSrc[MYsrcIndex+4];
		PixBuffer[Mytid3+5] = ImgSrc[MYsrcIndex+5];
	}
	__syncthreads();

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = PixBuffer[Mytid3];
	ImgDst[MYdstIndex + 1] = PixBuffer[Mytid3+1];
	ImgDst[MYdstIndex + 2] = PixBuffer[Mytid3+2];
	if(MYcol2 < Hpixels){
		ImgDst[MYdstIndex2] = PixBuffer[Mytid3+3];
		ImgDst[MYdstIndex2 + 1] = PixBuffer[Mytid3+4];
		ImgDst[MYdstIndex2 + 2] = PixBuffer[Mytid3+5];
	}

}
// each thread only flips 2 single pixel (R,G,B)
// use shared memory 3*1024*4 bytes, 1024*4 pixels, with less registers, put ScalerFactor outside the box, and 2D block
// Each kernel: uses Shared Memory (PixBuffer[]) to read in 12 Bytes
// (4 pixels). 4 pixels of new locations are calculated. 
// After that, they are written into Global Mem as 3 int's
// Horizontal resolution MUST BE A POWER OF 4.
__global__
void imrotate5(uch *ImgDst, ui *ImgSrc, ui Vpixels, ui Hpixels, ui RowBytes, ui RowInts, double cosRot, double sinRot, double ScaleFactor)
{	
	__shared__ ui PixBuffer[3072*2];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYtid3 = MYtid * 3;
	ui MYrow = blockIdx.y;
	/////  modify from here
	ui MYcolIndex = (MYbid*ThrPerBlk + MYtid)*3;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYOffset = MYrow * RowInts;
	ui MYsrcIndex = MYOffset + MYcolIndex;

	// ui MYcol = MYbid*ThrPerBlk + MYtid;
	// if (MYcol >= Hpixels) return;			// col out of range
	// ui MYsrcOffset = MYrow * RowBytes;
	// ui MYsrcIndex = MYsrcOffset + 3 * MYcol;

	PixBuffer[MYtid3] = ImgSrc[MYsrcIndex];
	PixBuffer[MYtid3+1] = ImgSrc[MYsrcIndex+1];
	PixBuffer[MYtid3+2] = ImgSrc[MYsrcIndex+2];
	__syncthreads();

	ui MYdstIndex2, MYdstIndex3, MYdstIndex4;
	////////////// find destination index of first pixel	
	ui MYcol;
	int X, Y, NewCol, NewRow;
	double newX, newY;
	// find first MYdstIndex
	MYcol = (ui)(MYcolIndex*1.3);
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
	ui MYdstOffset = NewRow*RowBytes;
	ui MYdstIndex = MYdstOffset + 3 * NewCol;
	///////////////	

	// find second MYdstIndex2
	MYcol = MYcol + 1;
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
	MYdstOffset = NewRow*RowBytes;
	MYdstIndex2 = MYdstOffset + 3 * NewCol;
	///////////////	

	// find third MYdstIndex3
	MYcol = MYcol + 1;
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
	MYdstOffset = NewRow*RowBytes;
	MYdstIndex3 = MYdstOffset + 3 * NewCol;
	///////////////	

	// find fourth MYdstIndex4
	MYcol = MYcol + 1;
	X=(double)MYcol-(double)(Hpixels/2);
	Y=(double)(Vpixels/2)-(double)MYrow;
	newX=cosRot*X-sinRot*Y;
	newY=sinRot*X+cosRot*Y;
	newX=newX*ScaleFactor;
	newY=newY*ScaleFactor;
	NewCol=((int) (newX+Hpixels/2));
	NewRow=Vpixels/2-(int)newY;
	MYdstOffset = NewRow*RowBytes;
	MYdstIndex4 = MYdstOffset + 3 * NewCol;
	///////////////	

	uch *BuffPtr = (uch*)(&PixBuffer[MYtid3]);

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = BuffPtr[0];
	ImgDst[MYdstIndex+1] = BuffPtr[1];
	ImgDst[MYdstIndex+2] = BuffPtr[2];
	ImgDst[MYdstIndex2] = BuffPtr[3];
	ImgDst[MYdstIndex2+1] = BuffPtr[4];
	ImgDst[MYdstIndex2+2] = BuffPtr[5];
	ImgDst[MYdstIndex3] = BuffPtr[6];
	ImgDst[MYdstIndex3+1] = BuffPtr[7];
	ImgDst[MYdstIndex3+2] = BuffPtr[8];
	ImgDst[MYdstIndex4] = BuffPtr[9];
	ImgDst[MYdstIndex4+1] = BuffPtr[10];
	ImgDst[MYdstIndex4+2] = BuffPtr[11];
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
	printf("\n Input File name: %17s  (%d x %d)   File Size=%lu\n", fn, 
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
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%lu\n", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}


int main(int argc, char **argv)
{
	// char			Flip = 'H';
	// float			totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	float 			totalKernelExecutionTime, tmpKernelExcutionTime;
	cudaError_t		cudaStatus, cudaStatus2;
	cudaEvent_t		time1, time2;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow;
	// ui	 			BlkPerRowInt;
	ui				ThrPerBlk, NumBlocks;
	// ui 				NB2, NB4, NB8, RowInts;
	ui				RowBytes, RowInts;
	cudaDeviceProp	GPUprop;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	ui				*GPUImg32;
	char			SupportedBlocks[100];
	// int				KernelNum=1;
	char			KernelName[255];
	double			RotAngle, deltaAngle;					// rotation angle
	int 			RotIter;
	int 			TotalIters;
	double 			cosRot, sinRot;
	int 			configuration;
	strcpy(ProgName, "imrotateG");
	if(argc!=5){
		printf("\n\nUsage: ./imrotateG infile outfile N Config");
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

	configuration = atoi(argv[4]);
	switch(configuration){
		case 1: ThrPerBlk=128;
				break;
		case 2: ThrPerBlk=128;
				break;
		case 3: ThrPerBlk=256;
				break;
		case 4: ThrPerBlk=512;
				break;
		case 5: ThrPerBlk=1024;
				break;				
	}

	TotalIters = atoi(argv[3]);
	

	RowBytes = (IPH * 3 + 3) & (~3);
	RowInts = RowBytes / 4;
	BlkPerRow = CEIL(IPH,ThrPerBlk);
	// BlkPerRowInt = CEIL(RowInts, ThrPerBlk);
	NumBlocks = IPV*BlkPerRow; 
	dim3 dimGrid2D2(BlkPerRow,		   ip.Vpixels);
	dim3 dimGrid2D4(CEIL(BlkPerRow,4), ip.Vpixels);

	printf("\nNum blocks: %d\n", NumBlocks);
	printf("\nThread per block: %d\n", ThrPerBlk);

	if(TotalIters > 30){
		printf("\nN is too large, should be less or equal to 30\n");
	}
	deltaAngle = 2*PI/float(TotalIters);
	printf("\nTotal iterations: %d\n", TotalIters);

	// iteration to find all images
	GPUImg32 = (ui *)GPUImg;


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

		RotAngle = (double)(RotIter-1)*deltaAngle;
		cosRot = cos(RotAngle);
		sinRot = sin(RotAngle);
		double H=(double)IPH;
		double V=(double)IPV;
		double Diagonal=sqrt(H*H+V*V);
		double ScaleFactor=(IPH>IPV) ? V/Diagonal : H/Diagonal;
		printf("\nRotation angle = %lf", RotAngle);

		cudaEventCreate(&time1);
		cudaEventCreate(&time2);
		cudaEventRecord(time1, 0); // record time1 in the first iteration
		switch(configuration){
			case 1: //printf("\n Running in kernel configuration 1\n");
					imrotate <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPV, IPH, BlkPerRow, RowBytes, cosRot, sinRot);
					strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
					break;
			case 2: // printf("\n Running in kernel configuration 2\n");
					// ThrPerBlk = 1024;
					imrotate2 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPV, IPH, RowBytes, cosRot, sinRot, ScaleFactor);
					strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
					break;
			case 3: //printf("\n Running in kernel configuration 3\n");
					// ThrPerBlk = 1024;
					imrotate3 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPV, IPH, RowBytes, cosRot, sinRot, ScaleFactor);
					strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
					break;
			case 4: printf("\n Running in kernel configuration 4\n");
					// ThrPerBlk = 1024;
					imrotate4 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPV, IPH, RowBytes, cosRot, sinRot, ScaleFactor);
					strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
					break;
			case 5: printf("\n Running in kernel configuration 5\n");
					// ThrPerBlk = 1024;
					imrotate5 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg, GPUImg32, IPV, IPH, RowBytes, RowInts, cosRot, sinRot, ScaleFactor);
					strcpy(KernelName, "imrotate : Each thread rotate 1 pixel. Computes everything.\n");
					break;
			default:printf("...... Kernel Number=%d ... NOT IMPLEMENTED .... \n", configuration);
					strcpy(KernelName, "*** NOT IMPLEMENTED ***");
					break;
		}

		cudaEventRecord(time2, 0); //record time2 in teh last iteration
		cudaEventSynchronize(time1);
		cudaEventSynchronize(time2);
		cudaEventElapsedTime(&tmpKernelExcutionTime, time1, time2);
		totalKernelExecutionTime += tmpKernelExcutionTime;

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




	printf("Total Kernel Execution    =%7.2f ms\n", totalKernelExecutionTime);

	
	// cudaEventCreate(&time1);
	// cudaEventCreate(&time2);
	// cudaEventCreate(&time3);
	// cudaEventCreate(&time4);

	// cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer

	// cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	

	// RowInts = RowBytes / 4;

	// BlkPerRowInt = CEIL(RowInts, ThrPerBlk);
	// BlkPerRowInt2 = CEIL(CEIL(RowInts,2), ThrPerBlk);

	// dim3 dimGrid2D(BlkPerRow,		   ip.Vpixels);
	// dim3 dimGrid2D2(CEIL(BlkPerRow,2), ip.Vpixels);
	// dim3 dimGrid2D4(CEIL(BlkPerRow,4), ip.Vpixels);
	// dim3 dimGrid2Dint(BlkPerRowInt,    ip.Vpixels);
	// dim3 dimGrid2Dint2(BlkPerRowInt2,  ip.Vpixels);





	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.

	// cudaEventRecord(time3, 0);


	// GPUDataTransfer = 2*IMAGESIZE;

	// Copy output (results) from GPU buffer to host (CPU) memory.

	// cudaEventRecord(time4, 0);

	// cudaEventSynchronize(time1);
	// cudaEventSynchronize(time2);
	// cudaEventSynchronize(time3);
	// cudaEventSynchronize(time4);

	// cudaEventElapsedTime(&totalTime, time1, time4);
	// cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	// cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	// cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	

	// printf("\n--------------------------------------------------------------------------\n");
	// printf("%s    ComputeCapab=%d.%d  [max %s blocks; %lu thr/blk] \n",
	// 	GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	// printf("--------------------------------------------------------------------------\n");
	// printf("%s %s %s %c %u %u  [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, Flip, ThrPerBlk, KernelNum, NumBlocks, BlkPerRow);
	// printf("--------------------------------------------------------------------------\n");
	// printf("%s\n",KernelName);
	// printf("--------------------------------------------------------------------------\n");
	// printf("CPU->GPU Transfer   =%7.2f ms  ...  %4ld MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
	// printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
	// printf("GPU->CPU Transfer   =%7.2f ms  ...  %4ld MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU)); 
	// printf("--------------------------------------------------------------------------\n");
	// printf("Total time elapsed  =%7.2f ms       %4ld MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2*IMAGESIZE+GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
	// printf("--------------------------------------------------------------------------\n\n");

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUImg);
	cudaFree(GPUCopyImg);
	// cudaEventDestroy(time1);
	// cudaEventDestroy(time2);
	// cudaEventDestroy(time3);
	// cudaEventDestroy(time4);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
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



