/*
Simplified (and improved ;) version of gpu_anim.h from
the book "CUDA by example" which could be found here:
http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
*/

#ifndef __GPU_BITMAP_H__
#define __GPU_BITMAP_H__

#define GL_GLEXT_PROTOTYPES

#include <cuda.h>
#include <cuda_gl_interop.h>


#include <GL/glx.h>
#include <GL/glext.h>

// should include glut.h here but surprisingly on my system
// glut.h includes freeglut_std.h but not freeglut_ext.h
// I need freeglut_ext.h for the prototype of glutCloseFunc
#include <GL/freeglut.h>

#include "util.h"

struct GPUBitmap;
// a static pointer needed fot the glut callbacks
static GPUBitmap *bitmap;


struct GPUBitmap {
	int width, height;
	void *userData;

	// common buffer
	GLuint glBuffer;
	cudaGraphicsResource *cudaBuffer;

	// user callbacks
	void (*userRenderCallback)(uchar4*, void*, int);
	void (*userCleanCallback)(void*);
	void (*userKeyCallback)(unsigned char, void*);

#ifdef REPORT_FPS
	cudaEvent_t start, stop;
#endif


	GPUBitmap(int w, int h, void *data = NULL, char const *title = "GPU bitmap") {
		width = w;
		height = h;
		userData = data;

		// choose device for CUDA and GL
		cudaDeviceProp prop;
		int dev;
		memset(&prop, 0, sizeof(prop));
		prop.major = 1;
		prop.minor = 0;
		HANDLE_CUDA_ERR(cudaChooseDevice(&dev, &prop));
		//HANDLE_CUDA_ERR(cudaGLSetGLDevice(dev));

		// init GLUT
		int foo = 0;
		char *bar = NULL;
		glutInit(&foo, &bar);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow(title);

		// set common buffer
		glGenBuffers(1, &glBuffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glBuffer);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB);
		HANDLE_CUDA_ERR(cudaGraphicsGLRegisterBuffer(&cudaBuffer, glBuffer, cudaGraphicsMapFlagsNone));
	}

	void clean() {
		HANDLE_CUDA_ERR(cudaGraphicsUnregisterResource(cudaBuffer));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &glBuffer);
#ifdef REPORT_FPS
		HANDLE_CUDA_ERR(cudaEventDestroy(start));
		HANDLE_CUDA_ERR(cudaEventDestroy(stop));
#endif
	}


	void animate(void (*renderCallback)(uchar4*, void*, int),
			void(*cleanCallback)(void *), void(*keyCallback)(unsigned char, void *) = NULL) {
		bitmap = this;
		userRenderCallback = renderCallback;
		userCleanCallback = cleanCallback;
		userKeyCallback = keyCallback;
		glutDisplayFunc(displayFunc);
		glutIdleFunc(idleFunc);
		glutKeyboardFunc(keyFunc);
		glutCloseFunc(closeFunc);
#ifdef REPORT_FPS
		HANDLE_CUDA_ERR(cudaEventCreate(&start));
		HANDLE_CUDA_ERR(cudaEventCreate(&stop));
		START_TIMER(start);
#endif
		glutMainLoop();
	}


	static void displayFunc() {
		glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glutSwapBuffers();
	}


	static void idleFunc() {
		static int ticks = 0;
		uchar4 *devPtr;
		size_t size;

		HANDLE_CUDA_ERR(cudaGraphicsMapResources(1, &(bitmap->cudaBuffer), NULL));
		HANDLE_CUDA_ERR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, bitmap->cudaBuffer));
		bitmap->userRenderCallback(devPtr, bitmap->userData, ++ticks);
		HANDLE_CUDA_ERR(cudaGraphicsUnmapResources(1, &(bitmap->cudaBuffer), NULL));
		glutPostRedisplay();

#ifdef REPORT_FPS
		if (ticks % 100 == 0) {
			float elapsedTime;
			STOP_TIMER(bitmap->start, bitmap->stop, elapsedTime);
			fprintf(stderr, "%.2f fps\n", 100 * 1000 / elapsedTime);
			START_TIMER(bitmap->start);
		}
#endif
	}

	static void keyFunc(unsigned char key, int x, int y) {
		if (bitmap->userKeyCallback) {
			bitmap->userKeyCallback(key, bitmap->userData);
		}
	}


	static void closeFunc() {
		bitmap->clean();
		if (bitmap->userCleanCallback)
			bitmap->userCleanCallback(bitmap->userData);
	}
};
#endif
