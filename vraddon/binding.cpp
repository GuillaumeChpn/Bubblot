#include <nan.h>
#include <Windows.h>
#include <Commctrl.h>
#include "3dconnexion/si.h"
#include "3dconnexion/siapp.h"

using namespace v8;
using namespace Nan;
using namespace std;

static int          giSpwFlag;    /* Flag 0/1 si liaison avec 3D Mouse Device */
static int          giSpwLibFlagInit; /* Flag 0/1 si lib 3dx deja initialisee */
static SiHdl        gSpwDevHdl;   /* Handle to 3D Mouse Device */

static int spw_init();
static void spw_close();
static int spw_checkEvent(MSG *msg, int *type_ev, int *key, int *loc_x, int *loc_y);
static int giSpnavX;
static int giSpnavY;
static int giSpnavZ;
static int giSpnavRx;
static int giSpnavRy;
static int giSpnavRz;
static SPWfloat32 giSpnavScale;
struct handle_data {
	unsigned long process_id;
	HWND best_handle;
	char *window_name;
};
static HWND gHwndDesktop;
static DWORD wndThreadId;
static UINT spwMessageId = 0;
static Nan::Callback* nanCallback = NULL;

#if _MSC_VER >= 1300    // for VC 7.0
// from ATL 7.0 sources
#ifndef _delayimp_h
extern "C" IMAGE_DOS_HEADER __ImageBase;
#endif
#endif

//#define DEBUG

HMODULE GetCurrentModule()
{
#if _MSC_VER < 1300    // earlier than .NET compiler (VC 6.0)
	// Here\"s a trick that will get you the handle of the module
	// you\"re running in without any a-priori knowledge:
	// http://www.dotnet247.com/247reference/msgs/13/65259.aspx

	MEMORY_BASIC_INFORMATION mbi;
	static int dummy;
	VirtualQuery(&dummy, &mbi, sizeof(mbi));

	return reinterpret_cast<HMODULE>(mbi.AllocationBase);

#else    // VC 7.0
	// from ATL 7.0 sources

	return reinterpret_cast<HMODULE>(&__ImageBase);
#endif
}



/************************************************************************/
/* 3DCONNEXION SPACE WARE START *****************************************/
/************************************************************************/

/*----------------------------------------------------------------------*/
/* NOM         : spw_init                                               */
/* FICHIER     : nt_wedge.c                                             */
/* VERSION     : Tell2014                                               */
/* CREATION    : 28/10/2013                                             */
/* CREE PAR    : Thierry Sipiere #040                                   */
/* MODIFIE PAR :                                                        */
/* FUNC GROUP  : 3DCONNEXION SPACE WARE                                 */
/*----------------------------------------------------------------------*/
/* DESCRIPTION : Initialise la connexion avec 3DCONNEXION SPACE WARE.   */
/*----------------------------------------------------------------------*/
/*                                                                      */
/* VARIABLES EXTERNES                                                   */
/* ------------------                                                   */
/*    gHwndDesktop = handle fenetre desktop                             */
/*                                                                      */
/* VALEUR DE RETOUR                                                     */
/* ----------------                                                     */
/*    0 = Erreur                                                        */
/*    1 = OK                                                            */
/*----------------------------------------------------------------------*/

static int spw_init()

{
	SiOpenData oData;                    /* OS Independent data to open ball  */
	int iNumDev;

	gSpwDevHdl = NULL;
	OutputDebugString("[vrviewer] Reset Handle Device 3dconnexion");


	if (gHwndDesktop == NULL) {
		OutputDebugString("[vrviewer] Handler Window null");
		return 0;
	}

	giSpwLibFlagInit = 0; /* Flag si lib 3dx deja init */

	if (!giSpwLibFlagInit) {
		iNumDev = SiGetNumDevices();
		if (iNumDev >= 0)  giSpwLibFlagInit = 1; /* lib 3dx deja init */
	}

	/*init the SpaceWare input library */
	if (!giSpwLibFlagInit)
	{
		if (SiInitialize() != SPW_NO_ERROR) {
			OutputDebugString("[vrviewer] SiInitialize failed");
			return 0;
		}
	}

	SiOpenWinInit(&oData, gHwndDesktop);    /* init Win. platform specific data  */

												 /* open data, which will check for device type and return the device handle
												 to be used by this function */
	if ((gSpwDevHdl = SiOpen("NodeJS", SI_ANY_DEVICE, SI_NO_MASK,
		SI_EVENT, &oData)) == NULL)
	{
		if (!giSpwLibFlagInit) {
			OutputDebugString("[vrviewer] Init failed terminating");
			SiTerminate();  /* called to shut down the SpaceWare input library */
		}
		OutputDebugString("[vrviewer] 3D Connexion Open Failed");
		return 0;
	}
	SiSetUiMode(SI_ALL_HANDLES, SI_UI_NO_CONTROLS); /* Config SoftButton Win Display */


	char log[255]; sprintf(log, "[vrviewer] SiOpen gSpwDevHdl = %d", gSpwDevHdl); OutputDebugString(log);


	return 1;
}

/*----------------------------------------------------------------------*/
/* NOM         : spw_close                                              */
/* FICHIER     : nt_wedge.c                                             */
/* VERSION     : Tell2014                                               */
/* CREATION    : 28/10/2013                                             */
/* CREE PAR    : Thierry Sipiere #040                                   */
/* MODIFIE PAR :                                                        */
/* FUNC GROUP  : 3DCONNEXION SPACE WARE                                 */
/*----------------------------------------------------------------------*/
/* DESCRIPTION : Termine la connexion avec 3DCONNEXION SPACE WARE.      */
/*----------------------------------------------------------------------*/

static void spw_close()

{
	if (!giSpwFlag) return;

	SiClose(gSpwDevHdl); /* closes device */
	if (!giSpwLibFlagInit)
		SiTerminate(); /* shut down the 3DxWare Input Library correctly */

	gSpwDevHdl = NULL;
	OutputDebugString("[vrviewer] Close Handle Device 3dconnexion");
	giSpwFlag = 0;
}

/*----------------------------------------------------------------------*/
/* NOM         : spw_calibrate                                          */
/* FICHIER     : nt_wedge.c                                             */
/* VERSION     : Tell2014                                               */
/* CREATION    : 14/11/2013                                             */
/* CREE PAR    : Serge Favre                                            */
/* MODIFIE PAR :                                                        */
/* FUNC GROUP  : 3DCONNEXION SPACE WARE                                 */
/*----------------------------------------------------------------------*/
/* DESCRIPTION : Recalibration de la souris 3D                          */
/*----------------------------------------------------------------------*/

extern void spw_calibrate()

{
	if (!giSpwFlag) return;

	SiRezero(gSpwDevHdl); /* calibrate device */
}

/*----------------------------------------------------------------------*/
/* NOM         : spw_checkEvent                                         */
/* FICHIER     : nt_wedge.c                                             */
/* VERSION     : Tell2014                                               */
/* CREATION    : 28/10/2013                                             */
/* CREE PAR    : Thierry Sipiere #040                                   */
/* MODIFIE PAR :                                                        */
/*----------------------------------------------------------------------*/
/* DESCRIPTION : Teste si un message Windows est un evenement Space Ware */
/*               et le traite.                                          */
/*----------------------------------------------------------------------*/
/*                                                                      */
/* PARAMETRES D\"ENTREE                                                  */
/* -------------------                                                  */
/*    msg     : Structure message Windows                               */
/*                                                                      */
/* PARAMETRES DE SORTIE                                                 */
/* --------------------                                                 */
/*    type_ev :                                                         */
/*    key     :                                                         */
/*    loc_x   :                                                         */
/*    loc_y   :                                                         */
/*                                                                      */
/* VALEUR DE RETOUR                                                     */
/* ----------------                                                     */
/*    0 = Ce n\"est pas un message Space Ware                            */
/*    1 = C\"est un message Space Ware                                   */
/*----------------------------------------------------------------------*/

static int  spw_checkEvent(MSG *msg, int *type_ev, int *key, int *loc_x, int *loc_y)
{
	int iNumButton;
	int /*SpwRetVal*/      ret;

	SiSpwEvent     Event;    /* SpaceWare Event */
	SiGetEventData EData;    /* SpaceWare Event Data */

							 /* init Window platform specific data for a call to SiGetEvent */
	SiGetEventWinInit(&EData, msg->message, msg->wParam, msg->lParam);
	/* check whether msg was a 3D mouse event and process it */
	ret = SiGetEvent(gSpwDevHdl, SI_AVERAGE_EVENTS, &EData, &Event);
	char log[255]; 
	#ifdef DEBUG
	sprintf(log, "[vrviewer] SiGetEvent ret = %d handle = %d", ret, gSpwDevHdl); OutputDebugString(log);
	#endif
	if (ret == SI_SKIP_EVENT) return 1;
	if (ret != SI_IS_EVENT) return 0;

	switch (Event.type)
	{
		/* 3D mouse motion event */
	case SI_MOTION_EVENT:
		giSpnavX = Event.u.spwData.mData[SI_TX];
		giSpnavY = Event.u.spwData.mData[SI_TY];
		giSpnavZ = Event.u.spwData.mData[SI_TZ];
		giSpnavRx = Event.u.spwData.mData[SI_RX];
		giSpnavRy = Event.u.spwData.mData[SI_RY];
		giSpnavRz = Event.u.spwData.mData[SI_RZ];
		SiSyncGetScaleOverall ( gSpwDevHdl, &giSpnavScale );

		*type_ev = 3;

		return 1;

		/* 3D mouse zero event */
	case SI_ZERO_EVENT:
		/* quand le mouvement s\"arrete */
		*type_ev = 3;
		*key = -1;
		return 1;

		/* 3D mouse button event */
	case SI_BUTTON_EVENT:
		if ((iNumButton = SiButtonPressed(&Event)) != SI_NO_BUTTON)
		{
			*type_ev = 3;
			*key = iNumButton;
		}
		sprintf(log, "[vrviewer] Button pressed iNumButton %d", iNumButton);
		OutputDebugString(log);
		return 1;

		/* Device change event */
	case SI_DEVICE_CHANGE_EVENT:
//		if (debug_mode) printf("SPW device change event\n");
		return 1;

	default:
//		if (debug_mode) printf("SPW UNKNOWN event\n");
		return 1;
	}

	return 1;
}


BOOL is_main_window(HWND handle)
{
	return GetWindow(handle, GW_OWNER) == (HWND)0 && IsWindowVisible(handle);
}

BOOL CALLBACK enum_windows_callback(HWND handle, LPARAM lParam)
{
	handle_data& data = *(handle_data*)lParam;
	unsigned long process_id = 0;
	DWORD thread_id = GetWindowThreadProcessId(handle, &process_id);
	char name[1000];
	GetWindowText(handle, name, 1000);
	char log[255];
	sprintf(log, "[vrviewer] name = %s process_id=%X thread_id=%X", name, process_id, thread_id);
	OutputDebugString(log);
	if (strcmp(name, data.window_name)) { //  || !is_main_window(handle)
		OutputDebugString("[vrviewer] Incorrect handle");
		return TRUE;
	}
	data.best_handle = handle;
	OutputDebugString("[vrviewer] Windows found");
	wndThreadId = thread_id;
	return FALSE;
}


HWND find_main_window(unsigned long process_id, char *name)
{
	handle_data data;
	data.process_id = process_id;
	data.best_handle = 0;
	data.window_name = name;
	EnumWindows(enum_windows_callback, (LPARAM)&data);
	return data.best_handle;
}

static int initSharedValue(char *szName, int len) {
	HANDLE hMapFile;
	hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,   // read/write access
		FALSE,                 // do not inherit the name
		szName);               // name of mapping object

	if (hMapFile == NULL)
	{
		hMapFile = CreateFileMapping(
			INVALID_HANDLE_VALUE,    // use paging file
			NULL,                    // default security
			PAGE_READWRITE,          // read/write access
			0,                       // maximum object size (high-order DWORD)
			len,					 // maximum object size (low-order DWORD)
			szName);                 // name of mapping object
		if (hMapFile == NULL) {
			char log[255]; sprintf(log, "[vrviewer] Error OpenFileMapping Error %d", GetLastError()); OutputDebugString(log);
			return 0;
		}
	}
	return 1;
}

static int writeSharedValue(char *szName, void *buf, int len) {
	HANDLE hMapFile;
	LPCTSTR pBuf;

	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,    // use paging file
		NULL,                    // default security
		PAGE_READWRITE,          // read/write access
		0,                       // maximum object size (high-order DWORD)
		len,					 // maximum object size (low-order DWORD)
		szName);                 // name of mapping object

	if (hMapFile == NULL)
	{
		char log[255]; sprintf(log, "[vrviewer] Error CreateFileMapping Error %d", GetLastError()); OutputDebugString(log);
		return 0;
	}
	pBuf = (LPTSTR)MapViewOfFile(hMapFile,   // handle to map object
		FILE_MAP_ALL_ACCESS, // read/write permission
		0,
		0,
		len);

	if (pBuf == NULL)
	{
		OutputDebugString("[vrviewer] Error MapViewOfFile");
		CloseHandle(hMapFile);
		return 0;
	}


#ifdef DEBUG
	char log[255]; sprintf(log, "[vrviewer] Write Shared Value %s of len %d", szName, len); OutputDebugString(log);
#endif

	CopyMemory((PVOID)pBuf, buf, len);

	UnmapViewOfFile(pBuf);

	CloseHandle(hMapFile);

	return 1;
}

static int readSharedValue(char *szName, void *buf, int len) {
	HANDLE hMapFile;

	hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,   // read/write access
		FALSE,                 // do not inherit the name
		szName);               // name of mapping object

	if (hMapFile == NULL)
	{
		char log[255]; sprintf(log, "[vrviewer] Error OpenFileMapping Error %d", GetLastError()); OutputDebugString(log);
		return 0;
	}

	void *pBuf = MapViewOfFile(hMapFile, // handle to map object
		FILE_MAP_ALL_ACCESS,  // read/write permission
		0,
		0,
		len);

	if (pBuf == NULL)
	{
		OutputDebugString("[vrviewer] Error MapViewOfFile");
		CloseHandle(hMapFile);
		return 0;
	}
	CopyMemory(buf, pBuf, len);

#ifdef DEBUG
	char log[255]; sprintf(log, "[vrviewer] Read Shared Value %s of len %d", szName, len); OutputDebugString(log);
#endif

	UnmapViewOfFile(pBuf);
	CloseHandle(hMapFile);
	return len;
}

LRESULT CALLBACK mainWndProc(
	_In_ int    nCode,
	_In_ WPARAM wParam,
	_In_ LPARAM lParam
) {
	char log[255];
	MSG *msg = (MSG *)lParam;
	if (spwMessageId == 0) {
		readSharedValue("Local\\spwMessageId", &spwMessageId, sizeof(UINT));
		sprintf(log, "[vrviewer] spwMessageId received %d", spwMessageId); OutputDebugString(log);
	}
	int type_ev = 0, key = 0, loc_x = 0, loc_y = 0;
#ifdef DEBUG
	sprintf(log, "[vrviewer] Message received hWnd = %X message = %d lParam = %d wParam = %d", msg->hwnd, msg->message, msg->lParam, msg->wParam); OutputDebugString(log);
#endif
	if (spwMessageId != 0 && msg->message != spwMessageId) {
		return CallNextHookEx(NULL, nCode, wParam, lParam);
	}
	else {
			writeSharedValue("Local\\spwMsg", msg, sizeof(MSG));
			#ifdef DEBUG
				sprintf(log, "[vrviewer] Message received hWnd = %X message = %d lParam = %d wParam = %d", msg->hwnd, msg->message, msg->lParam, msg->wParam);
			#endif
			OutputDebugString(log);
		}

	return CallNextHookEx(NULL, nCode, wParam, lParam);
}
/*
class SpwWorker : public AsyncWorker {
public:
	SpwWorker(Callback *callback, SiHdl hdl)
		: AsyncWorker(callback), hdl(hdl) {}
	~SpwWorker() {}

	void Execute() {
		MSG msg;
		readSharedValue("Local\\spwMsg", &msg, sizeof(msg));
		if (spw_checkEvent(&msg, &type_ev, &key, &loc_x, &loc_y)) {
		}
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;

		Local<Value> argv[] = { Null(), results };
		callback->Call(2, argv);
	}

private:
	SiHdl hdl;
	int type_ev, key, loc_x, loc_y;
};*/

NAN_METHOD(init3dConnexion) {
	spw_close();
	Nan::MaybeLocal<String> maybeWindowName = Nan::To<String>(info[0]);
	v8::Local<String> windowName;
	if (maybeWindowName.ToLocal(&windowName) == false) {
		Nan::ThrowError("Error converting first argument to String");
	}

	gHwndDesktop = find_main_window(GetCurrentProcessId(), *String::Utf8Value(windowName));
	char log[255];
	initSharedValue("Local\\spwMsg", sizeof(MSG));
	initSharedValue("Local\\spwMessageId", sizeof(UINT));
#ifdef DEBUG
	sprintf(log, "[vrviewer] Found windows handle %llX for process %llX", (LONGLONG)gHwndDesktop, (LONGLONG)GetCurrentProcessId());
	OutputDebugString(log);
	OutputDebugString("[vrviewer] Init 3d connexion");
#endif
	if (spw_init()) {
		OutputDebugString("[vrviewer] 3d connexion successfully initiated");
		if (!SetWindowsHookEx(WH_GETMESSAGE, mainWndProc, GetModuleHandle("vrviewer.node"), wndThreadId)) {
//			if (!SetWindowsHookEx(WH_GETMESSAGE, mainWndProc , GetCurrentModule(), wndThreadId)) {
			sprintf(log, "[vrviewer] GetLastError %d Thread_id %X", GetLastError(), GetCurrentThreadId());
			OutputDebugString(log);
			OutputDebugString("[vrviewer] error hooking windows");
		}
		OutputDebugString("[vrviewer] Hook set-up");
	}

}

NAN_METHOD(get3dEvent) {
	MSG msg;
	Nan::HandleScope scope;
	readSharedValue("Local\\spwMsg", &msg, sizeof(MSG));
	info.GetReturnValue().Set(Encode("", 0, UTF8));
	if (msg.message != 0) {
#ifdef DEBUG
		char log[255];	sprintf(log, "[vrviewer] get3dEvent hWnd = %X message = %d lParam = %d wParam = %d msgLen = %l", msg.hwnd, msg.message, msg.lParam, msg.wParam, sizeof(MSG)); OutputDebugString(log);
		OutputDebugString("[vrviewer] new message available");
#endif
		int type_ev, key, loc_x, loc_y;
		if (spw_checkEvent(&msg, &type_ev, &key, &loc_x, &loc_y)) {
#ifdef DEBUG
			OutputDebugString("[vrviewer] Received message");
#endif
			char result[255];
			if (spwMessageId == 0) {
				spwMessageId = msg.message;
#ifdef DEBUG
				sprintf(log, "[vrviewer] spwMessageId filtered %d", spwMessageId); OutputDebugString(log);
#endif
				writeSharedValue("Local\\spwMessageId", &spwMessageId, sizeof(UINT));
			}

			sprintf(result, "{\"typeEv\": \"%d\", \"key\": \"%d\", \"tx\": \"%d\", \"ty\": \"%d\", \"tz\": \"%d\", \"rx\": \"%d\", \"ry\": \"%d\", \"rz\": \"%d\", \"scale\": \"%f\"}",
				type_ev, key, giSpnavX, giSpnavY, giSpnavZ, giSpnavRx, giSpnavRy, giSpnavRz, giSpnavScale);
			info.GetReturnValue().Set(Encode(result, strlen(result), UTF8));
		}
		msg.message = 0;
		writeSharedValue("Local\\spwMsg", &msg, sizeof(MSG));
	}
}

NAN_METHOD(close3dConnexion) {
	spw_close();
}



NAN_MODULE_INIT(Init) {
	OutputDebugString("[vrviewer] Starting 3dconnexion module");
	Nan::Set(target, Nan::New("init3dConnexion").ToLocalChecked(),
		Nan::GetFunction(Nan::New<FunctionTemplate>(init3dConnexion)).ToLocalChecked());
	Nan::Set(target, Nan::New("get3dEvent").ToLocalChecked(),
		Nan::GetFunction(Nan::New<FunctionTemplate>(get3dEvent)).ToLocalChecked());
	Nan::Set(target, Nan::New("close3dConnexion").ToLocalChecked(),
		Nan::GetFunction(Nan::New<FunctionTemplate>(close3dConnexion)).ToLocalChecked());
}

NODE_MODULE(myaddon, Init)
