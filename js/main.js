/*jshint esversion: 6 */
angular.module('bubblot', []).controller('mainController', ['$scope', '$element', '$timeout', '$interval', function ($scope, $element, $timeout, $interval) {
    var mainVm = this,
        serial = '',
        digitalio,
        digitalIoState = 64,
        servo1,
        servo2,
        servo3,
        servo4,
        tilt1,
        tilt2,
        compass,
        latitude,
        divHorizon,
        divHorizon2,
        div3DConnexion,
        div3DConnexion2,
        computeStyle = '',
        transitionTimeout,
        captureTimeout,
        receivedPitch = false,
        receivedRoll = false,
        receivedCompass = false,
        spacerx = 50,
        spacery = 50,
        errmsg = new YErrorMsg(),
        spwDataMax = 500,
        VSP1Radius = 0,
        VSP1Angle = 0,
        VSP1AngleServo1 = 0,
        VSP1AngleServo2 = 0,
        VSP2Radius,
        VSP2Angle,
        VSP2AngleServo1,
        VSP2AngleServo2,
        VSP3Radius,
        VSP3Angle,
        VSP3AngleServo1,
        VSP3AngleServo2,
        VSP4Radius,
        VSP4Angle,
        VSP4AngleServo1,
        VSP4AngleServo2;

    $scope.leftData = {
        focusLeftIndex: 0,
        positionClass: ["focus-left", "left-one", "left-two", "left-three", "left-four", "left-five", "left-six"],
        focusOrigClass: "left-six",
        horizonPitch: 0,
        horizonRoll: 0,
        twistLeft: false,
        twistRight: false,
        threeDAngle: 30,
        turbidityRed: 10,
        turbidityBlue: 80,
        turbidityGreen: 80,
        magnetism: 0.5,
        magnetismXaxis: 30,
        magnetismYaxis: 50,
        alarmGround: false,
        computeVa: false,
        vaData: [],
        pumpOn: false,
        pumpPower: 0,
        filledOk: false,
        help: false,
        displayListEmo: false
    };
    $scope.rightData = {
        focusRightIndex: 0,
        positionClass: ["focus-right", "right-one", "right-two", "right-three", "right-four", "right-five", "right-six"],
        focusOrigClass: "right-six",
        depth: 0,
        security: 90,
        securityAlert: false,
        ballastState: 70,
        ballastFill: false,
        ballastEmpty: false,
        thrust: 0,
        thrustTopSwitchOn: true,
        thrustBottomSwitchOn: true,
        thrustDragOn: false,
        engine1Angle: -90,
        engine1Radius: 0,
        engine2Angle: -45,
        engine2Radius: 0,
        engine3Angle: 45,
        engine3Radius: 0,
        engine4Angle: 180,
        engine4Radius: 0,
        angleB1: 30,
        angleB2: 240,
        angleB3: 90,
        angleStorage: 30,
        angleNorth: 180,
        distancexToPump: 20, //change for each bubblot
        distanceyToPump: -20,
        spotlightIntensity: 75,
        foglightIntensity: 75,
        spotlightSwitchOn: false,
        foglightSwitchOn: false,
        cameraRec: false,
        cameraRecMenu: true,
        cameraPlay: false,
        help: false,
        sonarDistance: 100,
    };
    $scope.leftDataPump = {
        focusLeftIndex: 0,
        positionClass: ["focus-left", "left-one"],
        focusOrigClass: "left-two",
        gpsCompass: -15,
        localLat: 12.583967,
        localLong: 32.985734,
        targetLat: 46.123432,
        targetLong: 36.023563,
        horizonPitch: 0,
        horizonRoll: 0

    };
    $scope.rightDataPump = {
        focusRightIndex: 0,
        positionClass: ["focus-right", "right-one", "right-two"],
        focusOrigClass: "right-three",
        powerNeedleAngle: 0,
        thrust: 0,
        depth: 0.7,
        security: 40,
        securityAlert: false,
        ballastState: 50,
        isCtrl: false,
        help: false,
        pulsesDirectionInput: 0,
        pulsesDirectionMeasured: 0,
    };
    $scope.winderData = {
        mainControl: 0,
        winderControl1: 0,
        winderControl2: 0,
        winderControl3: 0,
        winderControl4: 0,
        winderLength1: 0,
        winderSpeed1: 0,
        pressure1: 0.93,
        minPressure1: 0.8,
        pressureAlert1: false,
        winderAlert1: false,
        reset1: false,
        winderLength2: 0,
        winderSpeed2: 0,
        pressure2: 0.97,
        minPressure2: 0.7,
        pressureAlert2: false,
        winderAlert2: false,
        reset2: false,
        winderLength3: 0,
        winderSpeed3: 0,
        pressure3: 0.95,
        minPressure3: 0.6,
        pressureAlert3: false,
        winderAlert3: false,
        reset3: false,
        winderLength4: 0,
        winderSpeed4: 0,
        pressure4: 0.9,
        minPressure4: 0.9,
        pressureAlert4: false,
        winderAlert4: false,
        reset4: false,
        railMode: false,
        railLength: 0,
        railAlert: false,
        help: false
    };
    $scope.mapData = {
        displayFe: false,
        displayPb: false,
        displayCu: false,
        displaySn: false,
        displayPath: true,
        displayVa: true,
        displayMovie: false,
        displayMagn: false,
        displayTurbi: false,
        extractingData: false,
        bubblots: [],
        bubblotCursor: 1,
        dates: [],
        dateCursor: [2017, 2, 17, 17, 20],
        latitudeCursor: null,
        longitudeCursor: null,
        depthCursor: 0,
        tempCursor: 0,
        dataxPump: [],
        datayPump: [],
        datax: new Array(3),
        datay: new Array(3),
        isVa: new Array(3),
        isFe: new Array(3),
        isPb: new Array(3),
        isCu: new Array(3),
        isSn: new Array(3),
        infoVa: [],
        isMovie: new Array(3),
        avgMagnetism: new Array(3),
        infoMagnetism: [],
        isTurbi: new Array(3),
        avgTurbi: new Array(3),
        infoTurbiRed: [],
        infoTurbiGreen: [],
        infoTurbiBlue: [],
        zoom: 1,
        xImage: 0,
        yImage: 0,
        xNew: 0,
        yNew: 0,
        xLast: 0,
        yLast: 0,
        vaCursor: false,
        turbiCursor: false,
        magnCursor: false,
        movieCursor: false,
        updatePanel: false,
        graphName: "",
        playData: false,
        calendarFwd: false,
        calendarRwd: false,
    };
    $scope.notifData = {
        bubblotSecurity: false,
        bubblotTwistLeft: false,
        bubblotTwistRight: false,
        audioBubblot: false,
        pumpSecurity: false,
        audioPump: false,
        winderPressure1: false,
        winderReset1: false,
        winderPressure2: false,
        winderReset2: false,
        winderPressure3: false,
        winderReset3: false,
        winderPressure4: false,
        winderReset4: false,
        audioWinder: false,
        recData: false,
        help: false
    };
    var bubblotYoctoModules = {
        yDigitalIO: null,
        yTilt_Roll: null,
        yTilt_Pitch: null,
        yCompass: null,
        yServo1_Camera: null,
        yServo1_VSPTopLeft_1: null,
        yServo1_VSPTopLeft_2: null,
        yServo1_VSPTopRight_1: null,
        yServo1_VSPTopRight_2: null,
        yServo2_Thrust: null,
        yServo2_VSPBottomLeft_1: null,
        yServo2_VSPBottomLeft_2: null,
        yServo2_VSPBottomRight_1: null,
        yServo2_VSPBottomRight_2: null,
        yMotorDC_pump: null,
        yColorLed_turbi: null,
        yKnob_turbi: null,
        yKnob_sonar: null,
        yRelay_elecMagnet: null,
        yRelay_pump: null,
        yPressure_Security: null,
        yPressure_Depth: null,
    }

    var serialBubblot = {
        yDigitalIO: 'Yocto-Maxi-IO',
        y3d: 'Yocto-3D',
        yGps: 'Yocto-Gps',
        yServo1: 'Yocto-Servo1',
        yServo2: 'Yocto-Servo2',
        yServo3: 'Yocto-Servo3',
        yRelay: 'Yocto-Relay',
        yMotorDC: 'Yocto-Motor-DC',
        yColor: 'Yocto-Color',
        yKnob: 'Yocto-Knob',
        yPressureSecurity: 'Yocto-Pressure2',
        yPressureDepth: 'Yocto-Pressure1',
    }

    var pumpYoctoModules = {
        yPressure_Security: null,
        yPressure_Depth: null,
        yDigitalIO: null,
        yGps_Longitude: null,
        yGps_Latitude: null,
        yPwmInput_encoder: null,
        yMotorDC_pumpDirection: null,
    }

    var serialPump = {
        yPwmInput: 'Yocto-PWM-Rx',
        yMotorDC: 'Yocto-Motor-DC',
        yDigitalIO: 'Yocto-Maxi-IO',
    }

    var winderYoctoModules = {
        yRelay_WinderDirection: null,
        yPwmOutput_WinderSpeed: null,
        yPwmInput_WinderLength: null
    }

    var serialWinder = {
        yRelay: 'Yocto-Relay',
        yPwmOutput: 'Yocto-PWM-Tx',
        yPwmInput: 'Yocto-PWM-Rx'
    }

    async function connectYoctoBubblot(ipaddress, serials, modules) {
        var YAPI = _yocto_api.YAPI;
        await YAPI.LogUnhandledPromiseRejections();
        await YAPI.DisableExceptions();
        // Setup the API to use the VirtualHub on local machine
        if (await YAPI.RegisterHub('http://' + ipaddress + ':4444', errmsg) != YAPI.SUCCESS) {
            console.log('Cannot contact VirtualHub on ' + ipaddress + ': ' + errmsg.msg);
            return;
        }
        //Connexion to 3D module
        //Connexion to tilt module
        modules.yTilt_Roll = YTilt.FindTilt(serials.y3d + ".tilt1");
        if (await modules.yTilt_Roll.isOnline()) {
            console.log('Using module ' + serials.y3d + ".tilt1");
            await modules.yTilt_Roll.registerValueCallback(computeRoll);
        }
        else {
            console.log("Can't find module " + serials.y3d + ".tilt1");
        }
        //Connextion to pitch module
        modules.yTilt_Pitch = YTilt.FindTilt(serials.y3d + ".tilt2");
        if (await modules.yTilt_Pitch.isOnline()) {
            console.log('Using module ' + serials.y3d + ".tilt2");
            await modules.yTilt_Pitch.registerValueCallback(computePitch);
        }
        else {
            console.log("Can't find module " + serials.y3d + ".tilt2");
        }
        //Connexion to compass module
        modules.yCompass = YCompass.FindCompass(serials.y3d + ".compass");
        if (await modules.yCompass.isOnline()) {
            console.log('Using module ' + serials.y3d + ".compass");
            await modules.yCompass.registerValueCallback(computeCompass);
        }
        else {
            console.log("Can't find module " + serials.y3d + ".compass");
        }
        //Connexion to Digital-IO module
        modules.yDigitalIO = YDigitalIO.FindDigitalIO(serials.yDigitalIO + ".digitalIO");
        if (await modules.yDigitalIO.isOnline()) {
            console.log('Using module ' + serials.yDigitalIO + ".digitalIO");
            await modules.yDigitalIO.set_portDirection(0xF8); //Set 3 inputs (0,1,2) and 4 outputs (3,4,5,6,7)
            await modules.yDigitalIO.set_portOpenDrain(0x07); //Set 3 open drain (0,1,2) and 4 no open drain (3,4,5,6,7)
            await modules.yDigitalIO.set_portPolarity(0x00);
            await modules.yDigitalIO.registerValueCallback(computeIO);
        }
        else {
            console.log("Can't find module " + serials.yDigitalIO + ".digitalIO");
        }
        //Connexion to motor DC module
        modules.yMotorDC_pump = YMotor.FindMotor(serials.yMotorDC + ".motor");
        if (await modules.yMotorDC_pump.isOnline()) {
            console.log('Using module ' + serials.yMotorDC + ".motor");
            await modules.yMotorDC_pump.set_drivingForce(0);
        }
        else {
            console.log("Can't find module " + serials.yMotorDC + ".motor");
        }

        //Connexion to relay module
        modules.yRelay_elecMagnet = YRelay.FindRelay(serials.yRelay + ".relay1");
        if (await modules.yRelay_elecMagnet.isOnline()) {
            console.log('Using module ' + serials.yRelay + ".relay1");
            await modules.yRelay_elecMagnet.set_state(true);
        }
        else {
            console.log("Can't find module " + serials.yRelay + ".relay1");
        }
        //Connexion to servo 1 module 
        //Servo motor for camera
        modules.yServo1_Camera = YServo.FindServo(serials.yServo1 + ".servo1");
        if (await modules.yServo1_Camera.isOnline()) {
            console.log('Using module ' + serials.yServo1 + ".servo1");
        }
        else {
            console.log("Can't find module " + serials.yServo1 + ".servo1");
        }
        //Servo motor 1 for top left propeller
        modules.yServo1_VSPTopLeft_1 = YServo.FindServo(serials.yServo1 + ".servo3");
        if (await modules.yServo1_VSPTopLeft_1.isOnline()) {
            console.log("Using module " + serials.yServo1 + ".servo3");
            await modules.yServo1_VSPTopLeft_1.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo1 + ".servo3");
        }
        //Servo motor 2 for top left propeller
        modules.yServo1_VSPTopLeft_2 = YServo.FindServo(serials.yServo1 + ".servo2");
        if (await modules.yServo1_VSPTopLeft_2.isOnline()) {
            console.log('Using module ' + serials.yServo1 + ".servo2");
            await modules.yServo1_VSPTopLeft_2.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo1 + ".servo2");
        }
        //Servo motor 1 for top right propeller
        modules.yServo1_VSPTopRight_1 = YServo.FindServo(serials.yServo1 + ".servo5");
        if (await modules.yServo1_VSPTopRight_1.isOnline()) {
            console.log('Using module ' + serials.yServo1 + ".servo5");
            await modules.yServo1_VSPTopRight_1.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo1 + ".servo5");
        }
        //Servo motor 2 for top right propeller
        modules.yServo1_VSPTopRight_2 = YServo.FindServo(serials.yServo1 + ".servo4");
        if (await modules.yServo1_VSPTopRight_2.isOnline()) {
            console.log('Using module ' + serials.yServo1 + ".servo4");
            await modules.yServo1_VSPTopRight_2.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo1 + ".servo4");
        }
        //Connexion to servo 2 module 
        //Servo motor for thrust
        modules.yServo2_Thrust = YServo.FindServo(serials.yServo2 + ".servo1");
        if (await modules.yServo2_Thrust.isOnline()) {
            console.log('Using module ' + serials.yServo2 + ".servo1");
            //await modules.yServo2_Thrust.set_positionAtPowerOn(1000)
            //servoModule = await modules.yServo2_Thrust.get_module();
            //await servoModule.saveToFlash();
            await modules.yServo2_Thrust.set_position(-1000);
        }
        else {
            console.log("Can't find module " + serials.yServo2 + ".servo1");
        }
        //Servo motor 1 for bottom left propeller
        modules.yServo2_VSPBottomLeft_1 = YServo.FindServo(serials.yServo2 + ".servo3");
        if (await modules.yServo2_VSPBottomLeft_1.isOnline()) {
            console.log("Using module " + serials.yServo2 + ".servo3");
            await modules.yServo2_VSPBottomLeft_1.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo2 + ".servo3");
        }
        //Servo motor 2 for bottom left propeller
        modules.yServo2_VSPBottomLeft_2 = YServo.FindServo(serials.yServo2 + ".servo2");
        if (await modules.yServo2_VSPBottomLeft_2.isOnline()) {
            console.log('Using module ' + serials.yServo2 + ".servo2");
            await modules.yServo2_VSPBottomLeft_2.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo2 + ".servo2");
        }
        //Servo motor 1 for bottom right propeller
        modules.yServo2_VSPBottomRight_1 = YServo.FindServo(serials.yServo2 + ".servo5");
        if (await modules.yServo2_VSPBottomRight_1.isOnline()) {
            console.log('Using module ' + serials.yServo2 + ".servo5");
            await modules.yServo2_VSPBottomRight_1.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo2 + ".servo5");
        }
        //Servo motor 2 for bottom right propeller
        modules.yServo2_VSPBottomRight_2 = YServo.FindServo(serials.yServo2 + ".servo4");
        if (await modules.yServo2_VSPBottomRight_2.isOnline()) {
            console.log('Using module ' + serials.yServo2 + ".servo4");
            await modules.yServo2_VSPBottomRight_2.set_position(0);
        }
        else {
            console.log("Can't find module " + serials.yServo2 + ".servo4");
        }
        //Servo motor 3 volt amp
        modules.yServo3_VoltAmp = YServo.FindServo(serials.yServo3 + ".servo1");
        if (await modules.yServo3_VoltAmp.isOnline()) {
            console.log('Using module ' + serials.yServo3 + ".servo1");
            await modules.yServo3_VoltAmp.set_position(700);
        }
        else {
            console.log("Can't find module " + serials.yServo3 + ".servo1");
        }
        //Connexion to knob module
        //Turbidity detection
        modules.yKnob_turbi = YAnButton.FindAnButton(serials.yKnob + ".anButton5");
        if (await modules.yKnob_turbi.isOnline()) {
            console.log('Using module ' + serials.yKnob + ".anButton5");
            await modules.yKnob_turbi.registerValueCallback(computeTurbiValue);
        }
        else {
            console.log("Can't find module " + serials.yKnob + ".anButton5");
        }
        //Depth detection
        modules.yKnob_sonar = YAnButton.FindAnButton(serials.yKnob + ".anButton1");
        if (await modules.yKnob_sonar.isOnline()) {
            console.log('Using module ' + serials.yKnob + ".anButton1");
            //modules.yKnob_sonar.registerValueCallback(computeSonarValue);
        }
        else {
            console.log("Can't find module " + serials.yKnob + ".anButton1");
        }
        //Reed relay for ground detection 
        modules.yAnButton_reed = YAnButton.FindAnButton(serials.yKnob + ".anButton4");
        if (await modules.yAnButton_reed.isOnline()) {
            console.log('Using module ' + serials.yKnob + ".anButton4");
            await modules.yAnButton_reed.registerValueCallback(computeReedValue);
        }
        else {
            console.log("Can't find module " + serials.yKnob + ".anButton4");
        }
        //Connexion to color module
        modules.yColorLed_turbi = YColorLed.FindColorLed(serials.yColor + ".colorLed1");
        if (await modules.yColorLed_turbi.isOnline()) {
            console.log('Using module ' + serials.yColor + ".colorLed1");
            await modules.yColorLed_turbi.set_rgbColor(0x000000);
        }
        else {
            console.log("Can't find module " + serials.yColor + ".colorLed1");
        }
        //Connexion to security pressure module
        modules.yPressure_Security = YPressure.FindPressure(serials.yPressureSecurity + ".pressure");
        if (await modules.yPressure_Security.isOnline()) {
            console.log('Using module ' + serials.yPressureSecurity + ".pressure");
            //modules.yPressure_Security.unmuteValueCallbacks()
            modules.yPressure_Security.set_reportFrequency("1/s");
            modules.yPressure_Security.registerTimedReportCallback(computePressureSecurity);
        }
        else {
            console.log("Can't find module " + serials.yPressureSecurity + ".pressure");
        }
        //Connexion to depth pressure module
        modules.yPressure_Depth = YPressure.FindPressure(serials.yPressureDepth + ".pressure");
        if (await modules.yPressure_Depth.isOnline()) {
            console.log('Using module ' + serials.yPressureDepth + ".pressure");
            //modules.yPressure_Depth.unmuteValueCallbacks()
            modules.yPressure_Depth.set_reportFrequency("1/s");
            modules.yPressure_Depth.registerTimedReportCallback(computePressureDepth);
        }
        else {
            console.log("Can't find module " + serials.yPressureDepth + ".pressure");
        }
    }
    async function connectYoctoPump(ipaddress, serials, modules) {
        var YAPI = _yocto_api.YAPI;
        await YAPI.LogUnhandledPromiseRejections();
        await YAPI.DisableExceptions();
        // Setup the API to use the VirtualHub on local machine
        if (await YAPI.RegisterHub('http://' + ipaddress + ':4444', errmsg) != YAPI.SUCCESS) {
            console.log('Cannot contact VirtualHub on ' + ipaddress + ': ' + errmsg.msg);
            return;
        }
        else {
            console.log("VirtualHub on " + ipaddress + " connected")
        }
        // by default use any connected module suitable for the demo
        //Connexion to PWM input module
        modules.yPwmInput_encoder = YQuadratureDecoder.FindQuadratureDecoder(serials.yPwmInput);
        if (await modules.yPwmInput_encoder.isOnline()) {
            console.log('Using module ' + serials.yPwmInput + ".pwmInput1");
            //await modules.yPwmInput_encoder.set_pwmReportMode(YPwmInput.PWMREPORTMODE_PWM_PULSECOUNT);
            //await modules.yPwmInput_encoder.resetCounter();
            await modules.yPwmInput_encoder.set_decoding(YQuadratureDecoder.DECODING_ON);
            await modules.yPwmInput_encoder.set_currentValue(0)
            await modules.yPwmInput_encoder.registerValueCallback(computeEncoder);
        }
        else {
            console.log("Can't find module " + serials.yPwmInput + ".pwmInput1");
        }
        //Connexion to motor DC module
        modules.yMotorDC_pumpDirection = YMotor.FindMotor(serials.yMotorDC + ".motor");
        if (await modules.yMotorDC_pumpDirection.isOnline()) {
            console.log('Using module ' + serials.yMotorDC + ".motor");
            await modules.yMotorDC_pumpDirection.set_drivingForce(0);
        }
        else {
            console.log("Can't find module " + serials.yMotorDC + ".motor");
        }
    }

    function connectYoctoWinder(ipaddress, serials, modules) {
        var YAPI = _yocto_api.YAPI;
        YAPI.LogUnhandledPromiseRejections().then(() => {
            return YAPI.DisableExceptions();
        }
        ).then(() => {
            // Setup the API to use the VirtualHub on local machine
            return YAPI.RegisterHub('http://' + ipaddress + ':4444', errmsg);
        }
        ).then(() => {
            // by default use any connected module suitable for the demo
            //Connexion to relay module
            modules.yRelay_WinderDirection = YRelay.FindRelay(serials.yRelay + ".relay1");
            modules.yRelay_WinderDirection.isOnline().then((onLine) => {
                if (onLine) {
                    console.log('Using module ' + serials.yRelay + ".relay1");
                }
                else {
                    console.log("Can't find module " + serials.yRelay + ".relay1");
                }
            })
        }
        ).then(() => {
            // by default use any connected module suitable for the demo
            //Connexion to PWM output module
            modules.yPwmOutput_WinderSpeed = YPwmOutput.FindPwmOutput(serials.yPwmOutput + ".pwmOutput1");
            modules.yPwmOutput_WinderSpeed.isOnline().then((onLine) => {
                if (onLine) {
                    console.log('Using module ' + serials.yPwmOutput + ".pwmOutput1");
                    modules.yPwmOutput_WinderSpeed.set_frequency(5000);
                    modules.yPwmOutput_WinderSpeed.set_enabled(_yocto_api.Y_ENABLED_TRUE);
                    modules.yPwmOutput_WinderSpeed.set_dutyCycle(0);
                }
                else {
                    console.log("Can't find module " + serials.yPwmOutput + ".pwmOutput1");
                }
            })
        }
        ).then(() => {
            // by default use any connected module suitable for the demo
            //Connexion to PWM input module
            modules.yPwmInput_WinderLength = YPwmInput.FindPwmInput(serials.yPwmInput + ".pwmInput1");
            modules.yPwmInput_WinderLength.isOnline().then((onLine) => {
                if (onLine) {
                    console.log('Using module ' + serials.yPwmInput + ".pwmInput1");
                    modules.yPwmInput_WinderLength.resetCounter();
                }
                else {
                    console.log("Can't find module " + serials.yPwmInput + ".pwmInput1");
                }
            });
        }
        );
    }
    var serialPortOpenOptions = {
        baudRate: 115200,
        dataBits: 8,
        parity: 'none',
        stopBits: 1,
    };
    //Create serialport for DropSens sensor
    var serialPort = new SerialPort('COM9', serialPortOpenOptions, function (err) { if (err) console.error('Error opening port'); });
    //Create serialport for Arduino
    var serialArduino = new SerialPort('COM3', serialPortOpenOptions, function (err) { if (err) console.error('Error opening port'); });    
    //Create parser to readline on Arduino
    var arduinoParser = null
    if (serialArduino) serialArduino.pipe(new Readline({ delimiter: '\n' }));
    var previousWinderSpeed1 = 0, previousWinderSpeed2 = 0, switchWinderDirection1 = false, stopWinderTime, stopWinderOk = true, winderDirection1 = true;
    var gamepadIndex = -1;
    async function init() {
        //Connect to Yocto module
        await connectYoctoBubblot("192.168.1.4", serialBubblot, bubblotYoctoModules);
        //connectYoctoWinder("192.168.1.228", serialWinder, winderYoctoModules);
        //connectYoctoWinder2("192.168.2.4", serialWinder, winderYoctoModules);
        //connectYoctoWinder3("192.168.3.4", serialWinder, winderYoctoModules);
        //connectYoctoWinder4("192.168.4.4", serialWinder, winderYoctoModules);
        //await connectYoctoPump("localhost", serialPump, pumpYoctoModules);
        setInterval(computeWinderLength, 1000);
        //setInterval(computeSonarValue, 250);

        //Connection to gampepad
        window.addEventListener("gamepadconnected", function (e) {
            var gp = navigator.getGamepads()[e.gamepad.index];
            console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
                gp.index, gp.id,
                gp.buttons.length, gp.axes.length);
            if (gp.id == "PC Game Controller        (Vendor: 11ff Product: 3331)") {
                gamepadIndex = gp.index;
                setTimeout(gamepadLoop, 500);
                //setInterval(gamepadLoop, 150);
            }
        });

        //Open function for DropSens sensor
        var writeToSerial = ['$V;', '$Ltech=01;'];
        var indexSerial = 0;
        var vaPotential, vaCurrent;
        var tempStringData = [], serialAllData = [], packageTransmitted = 0, flag = false;
        /*
        serialPort.on('open', function () {
            console.log('Serialport opened, writing ' + writeToSerial[indexSerial]);
            serialPort.write(writeToSerial[indexSerial]);
            indexSerial++;
        });

        //Data callback for DropsSens sensor 
        serialPort.on('data', function (data) {
            //console.log('Data received: ', data.length);
            //console.log(new Uint16Array(data));
            var serialStringData = String.fromCharCode.apply(null, new Uint16Array(data));
            console.log('Serialport : received')
            console.log(serialStringData);
            //Setting parameters
            if (serialStringData[serialStringData.length - 3] = ';' && indexSerial < writeToSerial.length) {
                console.log('Serialport: writing ' + writeToSerial[indexSerial]);
                serialPort.write(writeToSerial[indexSerial]);
                indexSerial++;
            }
            else if ($scope.leftData.computeVa) {
                if (packageTransmitted == 0 && serialStringData[serialStringData.length - 1].charCodeAt() == 10) {
                    packageTransmitted = 1;
                    console.log("Serialport: Computing VA measurement... ");
                }
                else if (packageTransmitted == 1) {
                    if (serialStringData[serialStringData.length - 4] != "E") {
                        tempStringData = tempStringData + serialStringData;
                        flag = true;
                    }
                    else if (serialStringData[serialStringData.length - 4] == "E" && flag) {
                        flag = false;
                        serialAllData = tempStringData + serialStringData;
                        tempStringData = [];
                        packageTransmitted = 2;
                        console.log("Serialport: VA measurement results");
                        console.log(serialAllData);
                    }
                    else if (serialStringData[serialStringData.length - 4] == "E") {
                        serialAllData = serialStringData;
                        packageTransmitted = 2;
                        console.log("Serialport: VA measurement results");
                        console.log(serialAllData);
                    }
                }
                if (packageTransmitted == 2) {
                    console.log("Serialport: Data");
                    for (var i = 0; i < (serialAllData.length - 5) / 25; i++) {
                        vaPotential = 0;
                        vaCurrent = 0;
                        for (j = 0; j < 4; j++) {
                            var asciiCodeData = serialAllData[i * 25 + j + 10].charCodeAt();
                            if (asciiCodeData <= 57 && asciiCodeData > 48) {
                                vaPotential = vaPotential + (asciiCodeData - 48) * Math.pow(16, 3 - j);
                            }
                            else if (asciiCodeData <= 70 && asciiCodeData >= 65) {
                                vaPotential = vaPotential + (asciiCodeData + 10 - 65) * Math.pow(16, 3 - j);
                            }
                        }
                        for (j = 0; j < 4; j++) {
                            var asciiCodeData = serialAllData[i * 25 + j + 2].charCodeAt();
                            if (asciiCodeData <= 57 && asciiCodeData > 48) {
                                vaCurrent = vaCurrent + (asciiCodeData - 48) * Math.pow(16, 3 - j);
                            }
                            else if (asciiCodeData <= 70 && asciiCodeData >= 65) {
                                vaCurrent = vaCurrent + (asciiCodeData + 10 - 65) * Math.pow(16, 3 - j);
                            }
                        }
                        console.log("Applied potential: " + vaPotential / 1000 + " V " + "Current on electrode 1: " + vaCurrent + " mV*2");
                    }
                    packageTransmitted = 0;
                    $scope.leftData.computeVa = false;
                }
            }
        });
        //Callback for Arduino port 
        serialArduino.on('open', function () {
            console.log("Arduino Micro Connected");
        });
        var indexArduino = -1;
        //Callback for Arduino Data
        arduinoParser.on('data', function (data) {
            //Start signal
            if(data.charCodeAt(0) == 38){
                indexArduino=0;
            }
            //Read X value of magnetism
            else if(indexArduino == 0){
                $scope.leftData.magnetismXaxis = data/10.0;
                indexArduino++;
            }
            //Read Y value of magnetism
            else if(indexArduino == 1){
                $scope.leftData.magnetismYaxis = data/10.0;
                indexArduino++;
            }
            //Read sonar value
            else if(indexArduino == 2){
                $scope.rightData.sonarDistance = (10000-data)/92.0;
                $scope.$apply();
                indexArduino=0;
            }
        });
        */
        //Compute measurement for DropSens sensor
        $scope.$watch('leftData.computeVa', async function (value) {
            if (value) {
                vaPotential = 0;
                console.log('serialPort writing $R;');
                //serialPort.write('$R;');
                if (bubblotYoctoModules.yServo3_VoltAmp) {
                    //await bubblotYoctoModules.yServo3_VoltAmp.move(0, 1000);
                    await bubblotYoctoModules.yServo3_VoltAmp.move(1000, 3000);
                    setTimeout(async function () {
                        await bubblotYoctoModules.yServo3_VoltAmp.move(700, 250);
                        setTimeout(async function () {
                            await bubblotYoctoModules.yServo3_VoltAmp.move(1000, 3000);
                            setTimeout(async function () {
                                await bubblotYoctoModules.yServo3_VoltAmp.move(700, 250);
                                setTimeout(async function () {
                                    await bubblotYoctoModules.yServo3_VoltAmp.move(1000, 3000);
                                    setTimeout(async function () {
                                        await bubblotYoctoModules.yServo3_VoltAmp.move(700, 250);
                                        setTimeout(async function () {
                                            await bubblotYoctoModules.yServo3_VoltAmp.move(1000, 3000);
                                            setTimeout(async function () {
                                                await bubblotYoctoModules.yServo3_VoltAmp.move(700, 250);
                                                $scope.leftData.computeVa = false;
                                                $scope.$apply()
                                            }, 9000)
                                        }, 750);
                                    }, 3500);
                                }, 750);
                            }, 3500);
                        }, 750);
                    }, 3500);
                }
            }
        });

        //Terminal command for starting HotSpot: netsh wlan start hostednetwork
        //require('nw.gui').Window.get().showDevTools(); alert('pause'); debugger;
        /*
        if (module3dconnexion) {
            module3dconnexion.init3dConnexion("Bubblot NW");
        }
        */
        transitionTimeout = $timeout(getValues, 500);
        //setInterval(getEvent, 200);
        //Keyboard shortcuts
        var isOne = false, isTwo = false, isThree = false, isFour = false, isCtrl = false;
        document.onkeydown = function (e) {
            var keyCode = e.keyCode;
            //F1 is pressed
            if (keyCode == 112) {
                //Show shortcut help
                $scope.notifData.help = true;
                $scope.leftData.help = true;
                $scope.rightData.help = true;
                $scope.rightDataPump.help = true;
                $scope.winderData.help = true;
                $scope.$apply();
            }
            //1 is pressed
            else if (keyCode == 49) {
                isOne = true;
            }
            //2 is pressed
            else if (keyCode == 50) {
                isTwo = true;
            }
            //3 is pressed
            else if (keyCode == 51) {
                isThree = true;
            }
            //4 is pressed
            else if (keyCode == 52) {
                isFour = true;
            }
            //Ctrl is pressed
            else if (keyCode == 17) {
                $scope.rightDataPump.isCtrl = true;
                isCtrl = true;
            }
        };
        document.onkeyup = function (e) {
            var keyCode = e.keyCode;
            //F1 is released
            if (keyCode == 112) {
                //Hide shortcut help
                $scope.notifData.help = false;
                $scope.leftData.help = false;
                $scope.rightData.help = false;
                $scope.rightDataPump.help = false;
                $scope.winderData.help = false;
                $scope.$apply();
            }
            //1 is released
            else if (keyCode == 49) {
                isOne = false;
            }
            //2 is released
            else if (keyCode == 50) {
                isTwo = false;
            }
            //3 is released
            else if (keyCode == 51) {
                isThree = false;
            }
            //4 is released
            else if (keyCode == 52) {
                isFour = false;
            }
            //Ctrl is released
            else if (keyCode == 17) {
                $scope.rightDataPump.isCtrl = false;
                isCtrl = false;
            }
            //S is released
            else if (keyCode == 83 && $('#winderDisplay').is(':visible')) {
                $scope.winderData.mainControl = 0;
                $scope.winderData.winderControl1 = 0;
                $scope.winderData.winderControl2 = 0;
                $scope.winderData.winderControl3 = 0;
                $scope.winderData.winderControl4 = 0;
                $scope.$apply();
            }
            //R is released
            else if (keyCode == 82 && $('#winderDisplay').is(':visible')) {
                $scope.winderData.winderControl1 = 0;
                $scope.winderData.winderControl2 = 0;
                $scope.winderData.winderControl3 = 0;
                $scope.winderData.winderControl4 = 0;
                $scope.$apply();
            }
        };
        //Scroll winder control with keyboard shortcuts
        var winderScreen = document.getElementById('winderDisplay');
        winderScreen.addEventListener("wheel", function () {
            var clientRect = winderScreen.getBoundingClientRect();
            var minX = clientRect.left + (clientRect.width / 5);
            var maxX = clientRect.right - (clientRect.width / 5);
            if (event.screenX > minX && event.screenX < maxX) {
                if (isOne && !$scope.winderData.railMode) {
                    if ($scope.winderData.winderControl1 <= 0.5 && $scope.winderData.winderControl1 >= -0.5) {
                        $scope.winderData.winderControl1 = $scope.winderData.winderControl1 - event.deltaY / 1000;
                        if ($scope.winderData.winderControl1 > 0.5) {
                            $scope.winderData.winderControl1 = 0.5;
                        }
                        else if ($scope.winderData.winderControl1 < -0.5) {
                            $scope.winderData.winderControl1 = -0.5;
                        }
                        $scope.$apply();
                    }
                }
                else if (isTwo && !$scope.winderData.railMode) {
                    if ($scope.winderData.winderControl2 <= 0.5 && $scope.winderData.winderControl2 >= -0.5) {
                        $scope.winderData.winderControl2 = $scope.winderData.winderControl2 - event.deltaY / 1000;
                        if ($scope.winderData.winderControl2 > 0.5) {
                            $scope.winderData.winderControl2 = 0.5;
                        }
                        else if ($scope.winderData.winderControl2 < -0.5) {
                            $scope.winderData.winderControl2 = -0.5;
                        }
                        $scope.$apply();
                    }
                }
                else if (isThree && !$scope.winderData.railMode) {
                    if ($scope.winderData.winderControl3 <= 0.5 && $scope.winderData.winderControl3 >= -0.5) {
                        $scope.winderData.winderControl3 = $scope.winderData.winderControl3 - event.deltaY / 1000;
                        if ($scope.winderData.winderControl3 > 0.5) {
                            $scope.winderData.winderControl3 = 0.5;
                        }
                        else if ($scope.winderData.winderControl3 < -0.5) {
                            $scope.winderData.winderControl3 = -0.5;
                        }
                        $scope.$apply();
                    }
                }
                else if (isFour && !$scope.winderData.railMode) {
                    if ($scope.winderData.winderControl4 <= 0.5 && $scope.winderData.winderControl4 >= -0.5) {
                        $scope.winderData.winderControl4 = $scope.winderData.winderControl4 - event.deltaY / 1000;
                        if ($scope.winderData.winderControl4 > 0.5) {
                            $scope.winderData.winderControl4 = 0.5;
                        }
                        else if ($scope.winderData.winderControl4 < -0.5) {
                            $scope.winderData.winderControl4 = -0.5;
                        }
                        $scope.$apply();
                    }
                }
                else if (!isCtrl) {
                    if ($scope.winderData.mainControl <= 0.5 && $scope.winderData.mainControl >= -0.5) {
                        $scope.winderData.mainControl = $scope.winderData.mainControl - event.deltaY / 5000;
                        if ($scope.winderData.mainControl > 0.5) {
                            $scope.winderData.mainControl = 0.5;
                        }
                        else if ($scope.winderData.mainControl < -0.5) {
                            $scope.winderData.mainControl = -0.5;
                        }
                        $scope.$apply();
                    }
                }
            }

        });
        var j = 0;
        $scope.$watch('leftData.pumpOn', function (value) {
            if (value && bubblotYoctoModules.yMotorDC_pump) {
                bubblotYoctoModules.yMotorDC_pump.drivingForceMove($scope.leftData.pumpPower * 100, $scope.leftData.pumpPower * 500);
            }
            else if (!value && bubblotYoctoModules.yMotorDC_pump) {
                bubblotYoctoModules.yMotorDC_pump.drivingForceMove(0, $scope.leftData.pumpPower * 500);
            }
        });
        $scope.$watch('leftData.pumpPower', function (value) {
            if ($scope.leftData.pumpOn && bubblotYoctoModules.yMotorDC_pump) {
                bubblotYoctoModules.yMotorDC_pump.set_drivingForce(value * 100);
            }
        });
        $scope.$watch('rightData.spotlightSwitchOn', function (value) {
            if (bubblotYoctoModules.yDigitalIO) {
                if (value) bubblotYoctoModules.yDigitalIO.set_bitState(6, 1);
                else bubblotYoctoModules.yDigitalIO.set_bitState(6, 0);
                if (value) bubblotYoctoModules.yDigitalIO.set_bitState(7, 1);
                else bubblotYoctoModules.yDigitalIO.set_bitState(7, 0);
            }
        });
        $scope.$watch('rightData.foglightSwitchOn', function (value) {
            if (bubblotYoctoModules.yDigitalIO) {
                
                if (value) bubblotYoctoModules.yDigitalIO.set_bitState(4, 1);
                else bubblotYoctoModules.yDigitalIO.set_bitState(4, 0);
                if (value) bubblotYoctoModules.yDigitalIO.set_bitState(5, 1);
                else bubblotYoctoModules.yDigitalIO.set_bitState(5, 0);
            }
        });
        $scope.$watch('leftData.threeDAngle', function (value) {
            if (bubblotYoctoModules.yServo1_Camera) {
                bubblotYoctoModules.yServo1_Camera.set_position((value - 38) / 16 * 2000);
            }
        });
        $scope.$watch('rightData.thrust', function (value) {
            if (bubblotYoctoModules.yServo2_Thrust) {
                bubblotYoctoModules.yServo2_Thrust.set_position((value * 2000 - 1000));
            }
        });
        $scope.$watch('winderData.railMode', function (value) {
            if (value) {
                $scope.notifData.winderPressure1 = false;
                $scope.notifData.winderPressure2 = false;
                $scope.notifData.winderPressure3 = false;
                $scope.notifData.winderPressure4 = false;
                $scope.winderData.winderSpeed3 = 0;
                $scope.winderData.winderSpeed4 = 0;
                $scope.winderData.winderAlert1 = false;
            }
            else {
                $scope.winderData.railAlert = false;
            }
            if (winderYoctoModules.yPwmInput_WinderLength) {
                winderYoctoModules.yPwmInput_WinderLength.resetCounter();
                previousWinderPulse = 1;
            }
        });
        $scope.$watch('winderData.mainControl', function (value) {
            if ($scope.winderData.railMode) {
                $scope.winderData.winderSpeed1 = value * 11;
                $scope.winderData.winderSpeed2 = value * 11;
            }
            else {
                $scope.winderData.winderSpeed1 = value * 10 + $scope.winderData.winderControl1;
                $scope.winderData.winderSpeed2 = value * 10 + $scope.winderData.winderControl2;
                $scope.winderData.winderSpeed3 = value * 10 + $scope.winderData.winderControl3;
                $scope.winderData.winderSpeed4 = value * 10 + $scope.winderData.winderControl4;
            }
        });
        $scope.$watch('winderData.winderControl1', function (value) {
            $scope.winderData.winderSpeed1 = $scope.winderData.mainControl * 10 + value;
        });
        $scope.$watch('winderData.winderControl2', function (value) {
            $scope.winderData.winderSpeed2 = $scope.winderData.mainControl * 10 + value;
        });
        $scope.$watch('winderData.winderControl3', function (value) {
            $scope.winderData.winderSpeed3 = $scope.winderData.mainControl * 10 + value;
        });
        $scope.$watch('winderData.winderControl4', function (value) {
            $scope.winderData.winderSpeed4 = $scope.winderData.mainControl * 10 + value;
        });
        $scope.$watch('winderData.winderSpeed1', function (value) {
            if (previousWinderSpeed2 * value < 0 && !switchWinderDirection1 && !stopWinderOk) {
                switchWinderDirection1 = true;
                //Stop de motor and wait 5s before switching direction
                if (winderYoctoModules.yPwmOutput_WinderSpeed) {
                    winderYoctoModules.yPwmOutput_WinderSpeed.set_dutyCycle(0);
                    clearTimeout(stopWinderTime);
                    setTimeout(winderDirectionTimeout, 4000);
                }
            }
            else if (!switchWinderDirection1) {
                if (winderYoctoModules.yRelay_WinderDirection) {
                    if (value > 0) {
                        winderYoctoModules.yRelay_WinderDirection.set_state(true);
                        winderDirection1 = true;
                        stopWinderOk = false;
                        clearTimeout(stopWinderTime);
                    }
                    else if (value < 0) {
                        winderYoctoModules.yRelay_WinderDirection.set_state(false);
                        winderDirection1 = false;
                        stopWinderOk = false;
                        clearTimeout(stopWinderTime);
                    }
                    else stopWinderTime = setTimeout(stopWinderTimeout, 4000);
                }
                if (winderYoctoModules.yPwmOutput_WinderSpeed) {
                    winderYoctoModules.yPwmOutput_WinderSpeed.set_dutyCycle(Math.abs(value) / 5.5 * 100);
                }
            }
            previousWinderSpeed2 = previousWinderSpeed1;
            previousWinderSpeed1 = value;
        });
        $scope.$watch('winderData.pressureAlert1', function (value) {
            $scope.notifData.winderPressure1 = value;
        });
        $scope.$watch('winderData.pressureAlert2', function (value) {
            $scope.notifData.winderPressure2 = value;
        });
        $scope.$watch('winderData.pressureAlert3', function (value) {
            $scope.notifData.winderPressure3 = value;
        });
        $scope.$watch('winderData.pressureAlert4', function (value) {
            $scope.notifData.winderPressure4 = value;
        });
        $scope.$watch('winderData.reset1', function (value) {
            $scope.notifData.winderReset1 = value;
        });
        $scope.$watch('winderData.reset2', function (value) {
            $scope.notifData.winderReset2 = value;
        });
        $scope.$watch('winderData.reset3', function (value) {
            $scope.notifData.winderReset3 = value;
        });
        $scope.$watch('winderData.reset4', function (value) {
            $scope.notifData.winderReset4 = value;
        });
        $scope.$watch('rightData.securityAlert', function (value) {
            $scope.notifData.bubblotSecurity = value;
        });
        $scope.$watch('rightData.ballastFill', function (value) {
            if (bubblotYoctoModules.yDigitalIO) {
                if (value) {
                    bubblotYoctoModules.yDigitalIO.set_bitState(3, 1);
                    bubblotYoctoModules.yRelay_elecMagnet.set_state(false);
                    $scope.rightData.ballastState = 100;
                }
                else {
                    bubblotYoctoModules.yDigitalIO.set_bitState(3, 0);
                    $scope.rightData.ballastState = 50;
                }
            }
        });
        $scope.$watch('rightData.ballastEmpty', function (value) {
            if (bubblotYoctoModules.yDigitalIO) {
                if (value) {
                    bubblotYoctoModules.yRelay_elecMagnet.set_state(true);
                    $scope.rightData.ballastState = 0;
                }
                else {
                }
            }
        });
        $scope.$watch('leftData.twistLeft', function (value) {
            $scope.notifData.bubblotTwistLeft = value;
        });
        $scope.$watch('leftData.twistRight', function (value) {
            $scope.notifData.bubblotTwistRight = value;
        });
        $scope.$watch('rightDataPump.securityAlert', function (value) {
            $scope.notifData.pumpSecurity = value;
        });
        var audioBubblot = document.getElementById("audio-alert-bubblot");
        $scope.$watch('notifData.audioBubblot', function (value) {
            if (value) {
                audioBubblot.play();
            }
            else {
                audioBubblot.pause();
                audioBubblot.currentTime = 0;
            }
        });
        var audioPump = document.getElementById("audio-alert-pump");
        $scope.$watch('notifData.audioPump', function (value) {
            if (value) {
                audioPump.play();
            }
            else {
                audioPump.pause();
                audioPump.currentTime = 0;
            }
        });
        var audioWinder = document.getElementById("audio-alert-winder");
        $scope.$watch('notifData.audioWinder', function (value) {
            if (value) {
                audioWinder.play();
            }
            else {
                audioWinder.pause();
                audioWinder.currentTime = 0;
            }
        });
        var intervalDataSave;
        print($scope.notifData.recData)
        $scope.$watch('notifData.recData', function (value) {
            if ($scope.notifData.recData) intervalDataSave = setInterval(saveData, 2000);
            else clearInterval(intervalDataSave);
        });
    }
    var button11Pressed = false, button4Pressed = false, button2Pressed = false, button5Pressed = false, button3Pressed = false, button1Pressed=false,motorButtonWasPressed=false,motorActive=false;
    var coeffRadius=60, kZ=1, multiplier=0;
    //Loop function to get joystick values
    async function gamepadLoop() {
        var gamepad = navigator.getGamepads ? navigator.getGamepads() : (navigator.webkitGetGamepads ? navigator.webkitGetGamepads : []);
        if (gamepad) {
            if (gamepad[gamepadIndex]) {
                var gp = gamepad[gamepadIndex];
                //If bubblot display
                if ($('#bubblotDisplay').is(':visible')) {
                    $scope.rightDataPump.pulsesDirectionInput = 0;
                    //Check if ground alarm
                    if ($scope.leftData.alarmGround) {
                        //Move up
                        $scope.rightData.engine1Angle = 180 + 135;
                        $scope.rightData.engine2Angle = 180 + 135;
                        $scope.rightData.engine3Angle = 0 + 135;
                        $scope.rightData.engine4Angle = 0 + 135;
                        $scope.rightData.engine1Radius = 60;
                        $scope.rightData.engine2Radius = $scope.rightData.engine1Radius;
                        $scope.rightData.engine3Radius = $scope.rightData.engine1Radius;
                        $scope.rightData.engine4Radius = $scope.rightData.engine1Radius;
                        bubblotYoctoModules.yServo2_Thrust.set_position(-500);
                    }
                   
                    else {
                        // Detect button number 1 pression, to switch from Translation to Rotation
                        if(gp.buttons[0].pressed){
                            button1Pressed=true;
                        }
                        else{
                            button1Pressed=false;
                        }
                        //Activate the thrust with button 2
                        if(gp.buttons[1].pressed){
                            motorButtonWasPressed=true;
                        }
                        else if (!gp.buttons[1].pressed && motorButtonWasPressed==true){
                            if(motorActive){
                                motorActive=false;
                            }
                            else{
                                motorActive=true;
                                console.log('motorActive');
                            }
                            motorButtonWasPressed=false;
                        }
                        // Increasing the speed until the maximum
                        if(gp.buttons[4].pressed && multiplier<0.99){
                            multiplier+=0.01;
                        }
                        // Decreasing the speed until the minimum
                        if(gp.buttons[2].pressed && multiplier>0){
                            multiplier-=0.01;
                        }
                        console.log('multiplier :',multiplier);
                        // associate the axes of the joystick :
                        x=-gp.axes[1];
                        y=gp.axes[0];
                        z=gp.axes[5]*90;
                        
                        // Avoid the noise from the joystick:
                        if(Math.abs(x)<0.3){x=0};
                        if(Math.abs(y)<0.3){y=0};
                        if(Math.abs(z/90)<0.4){z=0};

                        // Calculate amplitude
                        radius=Math.sqrt(Math.pow(x,2)+Math.pow(y,2));
                        // Calculate the angle on XY plance
                        if(radius==0){
                            angleXY=0;
                        }
                        else{
                            angleXY=(Math.atan2(1, 0)-Math.atan2(x,y)) * 180 / Math.PI;
                        }
                        //Get joystick position
                        cosXY=math.cos(math.unit(angleXY, 'deg'));
                        sinXY=math.sin(math.unit(angleXY, 'deg'));
                        cosZ=math.cos(math.unit(z, 'deg'));
                        sinZ=math.sin(math.unit(z, 'deg'));

                        if (button1Pressed) {
                            //Rotation mode if button 2 is pressed
                            //joystickPitch= TBD with propulsion
                            //joystickRoll = TBD
                            $scope.rightData.engine1Angle = math.matrix([radius*sinXY,radius*cosXY+kZ*sinZ,0]);
                            $scope.rightData.engine2Angle = math.matrix([-radius*sinXY,-radius*cosXY+kZ*sinZ,0]);
                            $scope.rightData.engine3Angle = math.matrix([-radius*sinXY,radius*cosXY-kZ*sinZ,0]);
                            $scope.rightData.engine4Angle =math.matrix([radius*cosXY+kZ*sinZ,radius*sinXY,0]);
                                
                        }
                        else{
                            joystickRoll=0;
                            joystickPitch=0;
                            $scope.rightData.engine1Angle = math.matrix([radius*sinXY-kZ*sinZ,-radius*cosXY,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine2Angle = math.matrix([-radius*sinXY-kZ*sinZ,radius*cosXY,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine3Angle = math.matrix([radius*sinXY+kZ*sinZ,radius*cosXY,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine4Angle =math.matrix([radius*cosXY,-radius*sinXY+kZ*sinZ,0]); // Not rotated (module already rotated)
                        }
                        $scope.rightData.engine1Radius = coeffRadius*radius
                        $scope.rightData.engine2Radius = $scope.rightData.engine1Radius;
                        $scope.rightData.engine3Radius = $scope.rightData.engine1Radius;
                        $scope.rightData.engine4Radius = $scope.rightData.engine1Radius;
                        regulation=false;
                        if(regulation){
                            ////////////////////
                            // Regulation PID //
                            ////////////////////
                            errRoll=RollPID.computeError(joystickRoll,horizonRoll);
                            RollPID.totError(errRoll);
                            rollRegulated= RollPID.output();
                            
    
                            errPitch=PitchPID.computeError(joystickPitch,horizonPitch);
                            PitchPID.totError(errPitch);
                            pitchRegulated= PitchPID.output();
    
                            regulationAngle= Math.atan2(-rollRegulated,pitchRegulated) * 180 / Math.PI;
                            if(regulationAngle<0){
                                regulationAngle=360+regulationAngle;
                            }
                            cosAngleRegulated=math.cos(math.unit(regulationAngle, 'deg'));
                            sinAngleRegulated=math.sin(math.unit(regulationAngle, 'deg'));
    
                            /////////
                            
                            $scope.rightData.engine1Angle = math.matrix([(1-alpha)*(radius*sinXY-kZ*sinZ)+alpha*sinAngleRegulated,(1-alpha)*(-radius*cosXY)+alpha*cosAngleRegulated,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine2Angle = math.matrix([(1-alpha)*(-radius*sinXY-kZ*sinZ)-alpha*sinAngleRegulated,(1-alpha)*(radius*cosXY)-alpha*cosAngleRegulated,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine3Angle = math.matrix([(1-alpha)*(radius*sinXY+kZ*sinZ)-alpha*sinAngleRegulated,(1-alpha)*(radius*cosXY)+alpha*cosAngleRegulated,0]); // rotated by 90° compared to previous coordinates
                            $scope.rightData.engine4Angle = math.matrix([(1-alpha)*(radius*cosXY)+alpha*cosAngleRegulated,(1-alpha)*(-radius*sinXY+kZ*sinZ)+alpha*sinAngleRegulated,0]);
            
                        }

                        if(radius !=0 || z>0 || regulation){ // Gives an angle only of the radius is not null or just active on z axis
                            VSP1Angle = (Math.atan2(1, 0)-Math.atan2($scope.rightData.engine1Angle._data[0],$scope.rightData.engine1Angle._data[1])) * 180 / Math.PI;
                            VSP2Angle = (Math.atan2(1, 0)-Math.atan2($scope.rightData.engine2Angle._data[0],$scope.rightData.engine2Angle._data[1])) * 180 / Math.PI;
                            VSP3Angle = (Math.atan2(1, 0)-Math.atan2($scope.rightData.engine3Angle._data[0],$scope.rightData.engine3Angle._data[1])) * 180 / Math.PI;
                            VSP4Angle = (Math.atan2(1, 0)-Math.atan2($scope.rightData.engine4Angle._data[0],$scope.rightData.engine4Angle._data[1])) * 180 / Math.PI;
                            if(radius== 0 && regulation){
                                radius=1;
                            }
                        }
                        else{
                            VSP1Angle=0;
                            VSP2Angle=0;
                            VSP3Angle=0;
                            VSP4Angle=0;
                        }
                        if(VSP1Angle<0){ // Follow the same angle system as the joystick (never negative)
                            VSP1Angle=360+VSP1Angle;
                        }
                        if(VSP2Angle<0){ // Follow the same angle system as the joystick (never negative)
                            VSP2Angle=360+VSP2Angle;
                        }
                        if(VSP3Angle<0){ // Follow the same angle system as the joystick (never negative)
                            VSP3Angle=360+VSP3Angle;
                        }
                        if(VSP4Angle<0){ // Follow the same angle system as the joystick (never negative)
                            VSP4Angle=360+VSP4Angle;
                        }
                        
                    }
                    //Button 4 pressed => reduce pump power
                    if (gp.buttons[3].pressed && !button2Pressed) {
                        $scope.leftData.pumpPower = (($scope.leftData.pumpPower - 0.0834) < 0 ? 0 : $scope.leftData.pumpPower - 0.0834);
                        button2Pressed = true;
                    }
                    //Button 4 released
                    else if (!gp.buttons[3].pressed) button2Pressed = false;
                    //Button 6 pressed => increase pump power
                    if (gp.buttons[5].pressed && !button4Pressed) {
                        $scope.leftData.pumpPower = (($scope.leftData.pumpPower + 0.0834) > 1 ? 1 : $scope.leftData.pumpPower + 0.0834);
                        button4Pressed = true;
                    }
                    //Button 6 released
                    else if (!gp.buttons[5].pressed) button4Pressed = false;
                    /*
                    //Button 4 pressed => reduce thrust power
                    if (gp.buttons[3].pressed && !button3Pressed) {
                        $scope.rightData.thrust = (($scope.rightData.thrust - 0.0834) < 0 ? 0 : $scope.rightData.thrust - 0.0834);
                        button3Pressed = true;
                    }
                    //Button 4 released
                    else if (!gp.buttons[3].pressed) button3Pressed = false;
                    //Button 6 pressed => increase thrust power
                    if (gp.buttons[5].pressed && !button5Pressed) {
                        $scope.rightData.thrust = (($scope.rightData.thrust + 0.0834) > 1 ? 1 : $scope.rightData.thrust + 0.0834);
                        button5Pressed = true;
                    }
                    //Button 6 released
                    else if (!gp.buttons[5].pressed) button5Pressed = false;
                    */

                    //Button 12 pressed => switch on pump 
                    if (gp.buttons[11].pressed) {
                        $scope.leftData.pumpOn = true;
                    }
                    //Button 12 released => switch off pump
                    else if (!gp.buttons[11].pressed) {
                        $scope.leftData.pumpOn = false;
                    }
                    //Axis 9 => switch on turbidity
                    if (gp.axes[9].toFixed(2) == 0.14) {
                        if (!turbiColorActivated) {
                            if (bubblotYoctoModules.yColorLed_turbi) {
                                await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0xFF0000);
                                amountTurbi = 0;
                                totalTurbi = 0;
                                turbiColorActivated = true;
                                setTimeout(computeTurbidityRed, 500);
                            }
                        }
                    }
                    //Axis 9 => switch off turbidity
                    else {
                        turbiColorActivated = false;
                    }
                }
                //If pump display
                else if ($('#pumpDisplay').is(':visible')) {
                    if (Math.abs(gp.axes[5]) > 0.3) {
                        $scope.rightDataPump.pulsesDirectionInput = parseInt(gp.axes[5] * 4000.0);
                    }
                    else {
                        $scope.rightDataPump.pulsesDirectionInput = 0;
                    }
                }
            }
            else {
                $scope.rightData.engine1Radius = 0;
                $scope.rightData.engine2Radius = 0;
                $scope.rightData.engine3Radius = 0;
                $scope.rightData.engine4Radius = 0;
                $scope.rightDataPump.pulsesDirectionInput = 0;
            }
        }
        else {
            $scope.rightData.engine1Radius = 0;
            $scope.rightData.engine2Radius = 0;
            $scope.rightData.engine3Radius = 0;
            $scope.rightData.engine4Radius = 0;
            $scope.rightDataPump.pulsesDirectionInput = 0;
        }
        $scope.$apply();
        //Calcul angles des servos en fonction du rayon et de l'angle des vecteurs VSP
        /*
        VSP1Angle = $scope.rightData.engine1Angle;
        VSP2Angle = $scope.rightData.engine2Angle;
        VSP3Angle = $scope.rightData.engine3Angle;
        VSP4Angle = $scope.rightData.engine4Angle;
        */
        VSP1Radius = $scope.rightData.engine1Radius;
        VSP2Radius = $scope.rightData.engine2Radius;
        VSP3Radius = $scope.rightData.engine3Radius;
        VSP4Radius = $scope.rightData.engine4Radius;
        

        VSP1AngleServo1 = -VSP1Radius / 45 * 10 * (-5.553 * Math.pow(10, -10) * Math.pow(VSP1Angle, 5) + 5.675 * Math.pow(10, -7) * Math.pow(VSP1Angle, 4)
            - 1.915 * Math.pow(10, -4) * Math.pow(VSP1Angle, 3) + 2.169 * Math.pow(10, -2) * Math.pow(VSP1Angle, 2) - 1.610 * Math.pow(10, -1) * VSP1Angle
            - 22.44);
        VSP1AngleServo2 = -VSP1Radius / 45 * 10 * (-5.415 * Math.pow(10, -10) * Math.pow(VSP1Angle, 5) + 4.225 * Math.pow(10, -7) * Math.pow(VSP1Angle, 4)
            - 9.197 * Math.pow(10, -5) * Math.pow(VSP1Angle, 3) + 2.847 * Math.pow(10, -3) * Math.pow(VSP1Angle, 2) + 2.585 * Math.pow(10, -1) * VSP1Angle
            + 31.42);
        VSP2AngleServo1 = -VSP2Radius / 45 * 10 * (-5.553 * Math.pow(10, -10) * Math.pow(VSP2Angle, 5) + 5.675 * Math.pow(10, -7) * Math.pow(VSP2Angle, 4)
            - 1.915 * Math.pow(10, -4) * Math.pow(VSP2Angle, 3) + 2.169 * Math.pow(10, -2) * Math.pow(VSP2Angle, 2) - 1.610 * Math.pow(10, -1) * VSP2Angle
            - 22.44);
        VSP2AngleServo2 = -VSP2Radius / 45 * 10 * (-5.415 * Math.pow(10, -10) * Math.pow(VSP2Angle, 5) + 4.225 * Math.pow(10, -7) * Math.pow(VSP2Angle, 4)
            - 9.197 * Math.pow(10, -5) * Math.pow(VSP2Angle, 3) + 2.847 * Math.pow(10, -3) * Math.pow(VSP2Angle, 2) + 2.585 * Math.pow(10, -1) * VSP2Angle
            + 31.42);
        VSP3AngleServo1 = -VSP3Radius / 45 * 10 * (-5.553 * Math.pow(10, -10) * Math.pow(VSP3Angle, 5) + 5.675 * Math.pow(10, -7) * Math.pow(VSP3Angle, 4)
            - 1.915 * Math.pow(10, -4) * Math.pow(VSP3Angle, 3) + 2.169 * Math.pow(10, -2) * Math.pow(VSP3Angle, 2) - 1.610 * Math.pow(10, -1) * VSP3Angle
            - 22.44);
        VSP3AngleServo2 = -VSP3Radius / 45 * 10 * (-5.415 * Math.pow(10, -10) * Math.pow(VSP3Angle, 5) + 4.225 * Math.pow(10, -7) * Math.pow(VSP3Angle, 4)
            - 9.197 * Math.pow(10, -5) * Math.pow(VSP3Angle, 3) + 2.847 * Math.pow(10, -3) * Math.pow(VSP3Angle, 2) + 2.585 * Math.pow(10, -1) * VSP3Angle
            + 31.42);
        VSP4AngleServo1 = -VSP4Radius / 45 * 10 * (-5.553 * Math.pow(10, -10) * Math.pow(VSP4Angle, 5) + 5.675 * Math.pow(10, -7) * Math.pow(VSP4Angle, 4)
            - 1.915 * Math.pow(10, -4) * Math.pow(VSP4Angle, 3) + 2.169 * Math.pow(10, -2) * Math.pow(VSP4Angle, 2) - 1.610 * Math.pow(10, -1) * VSP4Angle
            - 22.44);
        VSP4AngleServo2 = -VSP4Radius / 45 * 10 * (-5.415 * Math.pow(10, -10) * Math.pow(VSP4Angle, 5) + 4.225 * Math.pow(10, -7) * Math.pow(VSP4Angle, 4)
            - 9.197 * Math.pow(10, -5) * Math.pow(VSP4Angle, 3) + 2.847 * Math.pow(10, -3) * Math.pow(VSP4Angle, 2) + 2.585 * Math.pow(10, -1) * VSP4Angle
            + 31.42);
        //Moving servo motor to orientate propellers
        _yocto_api.YAPI.HandleEvents();
        if (bubblotYoctoModules.yServo1_VSPTopLeft_1) {
            await bubblotYoctoModules.yServo1_VSPTopLeft_1.set_position(Math.round(VSP1AngleServo1));
        }
        if (bubblotYoctoModules.yServo1_VSPTopLeft_2) {
            await bubblotYoctoModules.yServo1_VSPTopLeft_2.set_position(Math.round(VSP1AngleServo2));
        }
        if (bubblotYoctoModules.yServo1_VSPTopRight_1) {
            await bubblotYoctoModules.yServo1_VSPTopRight_1.set_position(Math.round(VSP2AngleServo1));
        }
        if (bubblotYoctoModules.yServo1_VSPTopRight_2) {
            await bubblotYoctoModules.yServo1_VSPTopRight_2.set_position(Math.round(VSP2AngleServo2));
        }
        if (bubblotYoctoModules.yServo2_VSPBottomLeft_2) {
            await bubblotYoctoModules.yServo2_VSPBottomLeft_2.set_position(Math.round(VSP3AngleServo2));
        }
        if (bubblotYoctoModules.yServo2_VSPBottomLeft_1) {
            await bubblotYoctoModules.yServo2_VSPBottomLeft_1.set_position(Math.round(VSP3AngleServo1));
        }
        if (bubblotYoctoModules.yServo2_VSPBottomRight_1) {
            await bubblotYoctoModules.yServo2_VSPBottomRight_1.set_position(Math.round(VSP4AngleServo1));
        }
        if (bubblotYoctoModules.yServo2_VSPBottomRight_2) {
            await bubblotYoctoModules.yServo2_VSPBottomRight_2.set_position(Math.round(VSP4AngleServo2));
        }
        if (bubblotYoctoModules.yServo2_Thrust && motorActive) {
            bubblotYoctoModules.yServo2_Thrust.set_position((multiplier * 2000 - 1000));
        }
        else{
            bubblotYoctoModules.yServo2_Thrust.set_position(-1000);
        }
        if (pumpYoctoModules.yMotorDC_pumpDirection) {
            await pumpYoctoModules.yMotorDC_pumpDirection.set_drivingForce(($scope.rightDataPump.pulsesDirectionInput - $scope.rightDataPump.pulsesDirectionMeasured) / 8000 * 80);
        }
        setTimeout(gamepadLoop, 200);
    }

    //Computer the length of the winder
    var previousWinderPulse = 1;
    function computeWinderLength() {
        //Computation of the length for rail mode
        if ($scope.winderData.railMode && winderYoctoModules.yPwmInput_WinderLength) {
            winderYoctoModules.yPwmInput_WinderLength.get_pulseCounter().then((value) => {
                //Compare speed with approximation
                if (value - previousWinderPulse < 4.3 * (Math.abs($scope.winderData.winderSpeed1) / 5.5 * 100) - 30 ||
                    value - previousWinderPulse > 4.3 * (Math.abs($scope.winderData.winderSpeed1) / 5.5 * 100) + 30) {
                    //Speed not as expected    
                    $scope.winderData.railAlert = true;
                }
                else {
                    //Speed ok
                    $scope.winderData.railAlert = false;
                }
                //Compute rail length
                if (!stopWinderOk) {
                    if (winderDirection1) $scope.winderData.railLength = $scope.winderData.railLength + (value - previousWinderPulse) / 100;
                    else $scope.winderData.railLength = $scope.winderData.railLength - (value - previousWinderPulse) / 100;
                    if ($scope.winderData.railLength < 0) {
                        $scope.winderData.railLength = 0;
                    }
                }
                previousWinderPulse = value;
                if (value > 10000) {
                    winderYoctoModules.yPwmInput_WinderLength.resetCounter();
                    previousWinderPulse = 1;
                }
                $scope.$apply();
            });
        }
        //Computation of the length for winder mode
        else if (!$scope.winderData.railMode && winderYoctoModules.yPwmInput_WinderLength) {
            winderYoctoModules.yPwmInput_WinderLength.get_pulseCounter().then((value) => {
                //Compare speed with approximation
                if (value - previousWinderPulse < 4.3 * (Math.abs($scope.winderData.winderSpeed1) / 5.5 * 100) - 30 ||
                    value - previousWinderPulse > 4.3 * (Math.abs($scope.winderData.winderSpeed1) / 5.5 * 100) + 30) {
                    //Speed not as expected    
                    $scope.winderData.winderAlert1 = true;
                }
                else {
                    //Speed ok
                    $scope.winderData.winderAlert1 = false;
                }
                //Compute winder length
                if (!stopWinderOk) {
                    if (winderDirection1) $scope.winderData.winderLength1 = $scope.winderData.winderLength1 + (value - previousWinderPulse) / 100;
                    else $scope.winderData.winderLength1 = $scope.winderData.winderLength1 - (value - previousWinderPulse) / 100;
                    if ($scope.winderData.winderLength1 < 0) {
                        $scope.winderData.winderLength1 = 0;
                    }
                }
                previousWinderPulse = value;
                if (value > 10000) {
                    winderYoctoModules.yPwmInput_WinderLength.resetCounter();
                    previousWinderPulse = 1;
                }
                $scope.$apply();
            });
        }
    }
    //Functions to convert hex to binary to get IO states
    var ConvertBase = function (num) {
        return {
            from: function (baseFrom) {
                return {
                    to: function (baseTo) {
                        return parseInt(num, baseFrom).toString(baseTo);
                    }
                };
            }
        };
    };
    ConvertBase.hex2bin = function (num) {
        return ConvertBase(num).from(16).to(2);
    };
    function computeIO(object, value) {
        valueBin = ConvertBase.hex2bin(value[1]);
        if ((valueBin & 0b0100) == 8) {
            //$scope.rightData.ballastState = 0;
            //$scope.rightData.ballastEmpty = false;
        }
        else if ((valueBin & 0b0010) == 2) {
            //$scope.rightData.ballastState = 100;
            //$scope.rightData.ballastFill = false;
        }
        else {
            //$scope.rightData.ballastState = 50;
        }
        $scope.$apply();
    }
    function getValues() {
        if (receivedPitch && receivedRoll) {
            receivedPitch = false;
            receivedRoll = false;
            $scope.$apply();
        }
    }
    function computePitch(object, value) {
        $scope.leftDataPump.horizonPitch = value;
        $scope.leftData.horizonPitch = value;
        if (!receivedPitch) {
            receivedPitch = true;
            if (receivedRoll) {
                getValues();
            }
        }
    }
    function computeRoll(object, value) {
        $scope.leftDataPump.horizonRoll = value;
        $scope.leftData.horizonRoll = value;
        if (!receivedRoll) {
            receivedRoll = true;
            if (receivedPitch) {
                getValues();
            }
        }
    }
    function computeCompass(object, value) {
        $scope.leftDataPump.gpsCompass = value;
        $scope.$apply();
    }
    function computeEncoder(object, value) {
        $scope.rightDataPump.pulsesDirectionMeasured = value;
    }
    function computePressureSecurity(object, YMeasure) {
        $scope.rightData.security = ((YMeasure._avgVal / 1000 - 1)) / 0.1 * 100;
        if ($scope.rightData.security > 100) $scope.rightData.security = 100;
        else if ($scope.rightData.security < 0) $scope.rightData.security = 0;
        $scope.$apply();
        /*
        if(YMeasure._avgVal/1000 - ($scope.rightData.depth*12 + 1) > 0.4) $scope.rightData.security = 100;
        else if(YMeasure._avgVal/1000 - ($scope.rightData.depth*12 + 1) < 0.2) $scope.rightData.security = 0;
        else $scope.rightData.security = (YMeasure._avgVal/1000 - ($scope.rightData.depth*12 + 1)-0.2)/0.2*100;
        $scope.$apply();
        */
    }
    function computePressureDepth(object, YMeasure) {
        /*
        if(YMeasure._avgVal < 1000) $scope.rightData.depth = 0;
        else $scope.rightData.depth = (YMeasure._avgVal/1000 - 1)/12;
        */
        $scope.rightData.depth = ((YMeasure._avgVal / 1000 - 1)) / 0.1;
        if ($scope.rightData.depth > 1) $scope.rightData.depth = 1;
        else if ($scope.rightData.depth < 0) $scope.rightData.depth = 0;
        $scope.$apply();
    }
    function computeLatitude(object, value) {
    }
    function computeTemp(object, value) {
    }
    var amountTurbi = 0, totalTurbi = 0, turbidity = 1000.0, lastTurbi = 1000.0;
    async function computeTurbiValue(object, value) {
        totalTurbi = totalTurbi + parseInt(value);
        amountTurbi++;
        lastTurbi = parseInt(value);
    }
    var timerReed = null;
    //Reed relai to detect ground
    async function computeReedValue(object, value) {
        //If no ground
        if (value > 500) {
            if ($scope.leftData.alarmGround) {
                timerReed = setTimeout(function () {
                    $scope.leftData.alarmGround = false;
                    bubblotYoctoModules.yServo2_Thrust.set_position(parseFloat($scope.rightData.thrust) * 2000 - 1000);
                }, 2000);
            }
        }
        //If ground
        else {
            $scope.leftData.alarmGround = true;
            if (timerReed) {
                clearTimeout(timerReed);
                timerReed = null;
            }
        }
    }
    async function computeTurbidityRed() {
        if (amountTurbi == 0) $scope.leftData.turbidityRed = (1 - lastTurbi / 1000.0) / 3.0;
        else $scope.leftData.turbidityRed = (1 - parseFloat(totalTurbi) / amountTurbi / 1000.0) / 3.0;
        if (turbiColorActivated) {
            await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0x00FF00);
            setTimeout(computeTurbidityGreen, 500);
            totalTurbi = 0, amountTurbi = 0;
        }
        else await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0x000000);
    }
    async function computeTurbidityGreen() {
        if (amountTurbi == 0) $scope.leftData.turbidityGreen = (1 - lastTurbi / 1000.0) / 3.0;
        else $scope.leftData.turbidityGreen = (1 - parseFloat(totalTurbi) / amountTurbi / 1000.0) / 3.0;
        if (turbiColorActivated) {
            await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0x0000FF);
            setTimeout(computeTurbidityBlue, 500);
            totalTurbi = 0, amountTurbi = 0;
        }
        else await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0x000000);
    }
    async function computeTurbidityBlue() {
        if (amountTurbi == 0) $scope.leftData.turbidityBlue = (1 - lastTurbi / 1000.0) / 3.0;
        else $scope.leftData.turbidityBlue = (1 - parseFloat(totalTurbi) / amountTurbi / 1000.0) / 3.0;
        if (turbiColorActivated) {
            await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0xFF0000);
            setTimeout(computeTurbidityRed, 500);
            totalTurbi = 0, amountTurbi = 0;
        }
        else await bubblotYoctoModules.yColorLed_turbi.set_rgbColor(0x000000);
    }
    function stopWinderTimeout() {
        stopWinderOk = true;
    }
    function winderDirectionTimeout() {
        if (winderYoctoModules.yRelay_WinderDirection && $scope.winderData.winderSpeed1 > 0) {
            winderYoctoModules.yRelay_WinderDirection.set_state(true);
            winderDirection1 = true;
        }
        else if (winderYoctoModules.yRelay_WinderDirection) {
            winderYoctoModules.yRelay_WinderDirection.set_state(false);
            winderDirection1 = false;
        }
        winderYoctoModules.yPwmOutput_WinderSpeed.set_dutyCycle(Math.abs($scope.winderData.winderSpeed1) / 5.5 * 100);
        switchWinderDirection1 = false;
    }
    var allScreen = document.getElementById('bubblotDisplay');
    allScreen.addEventListener('mousedown', dragDirection);
    var clientRect = allScreen.getBoundingClientRect();
    var margeX = clientRect.width / 5;
    var margeY = clientRect.height / 5;
    var minX = clientRect.left + margeX;
    var maxX = clientRect.right - margeX;
    var startX, startY, x, y;
    function dragDirection(event) {
        startX = event.screenX;
        startY = event.screenY;
        if (startX > minX && startX < maxX) {
            allScreen.addEventListener('mousemove', mousemove);
            allScreen.addEventListener('mouseup', mouseup);
        }
    }
    function mousemove(event) {
        x = event.screenX - startX;
        y = event.screenY - startY;
        if (x > margeX) {
            x = margeX;
        }
        if (x < -margeX) {
            x = -margeX;
        }
        if (y > margeY) {
            y = margeY;
        }
        if (y < -margeY) {
            y = -margeY;
        }
        bubblotDeltaY = x / margeX;
        bubblotDeltaX = -y / margeY;
    }
    function mouseup() {
        allScreen.removeEventListener('mousemove', mousemove);
        allScreen.removeEventListener('mouseup', mouseup);
    }

    var counter1 = 0, counter2 = 0, counter3 = 0;
    //Save the data in the database
    function saveData() {
        var date = new Date(), year, month, day, hours, minutes, seconds;
        year = date.getFullYear();
        month = date.getMonth() + 1;
        day = date.getDate();
        hours = date.getHours();
        minutes = date.getMinutes();
        seconds = date.getSeconds();

        $scope.rightData.distancexToPump = $scope.rightData.distancexToPump + 25 * Math.random() - 10; //Change for each bubblot
        $scope.rightData.distanceyToPump = $scope.rightData.distanceyToPump - 25 * Math.random() + 10;

        var fe = false, pb = false, cu = false, sn = false, isVa = false, isMovie = false;
        counter1++;
        counter2++;
        counter3++;
        var dataVa = [];
        if (counter1 == 6) {
            var va = 6 * Math.random();
            if (va <= 1) {
                isVa = true;
            }
            else if (va > 1 && va <= 2) {
                pb = true;
                isVa = true;
            }
            else if (va > 3 && va <= 4) {
                cu = true;
                isVa = true;
            }
            else if (va > 4 && va <= 5) {
                sn = true;
                isVa = true;
            }
            else {
                fe = true;
                isVa = true;
            }
            for (var i = 0; i <= 15; i++) {
                var obj = { x: 2 * i, y: 20 * Math.random() };
                dataVa.push(obj);
            }
            counter1 = 0;
        }
        if (counter2 == 20) {
            isMovie = true;
            counter2 = 0;
        }
        var isTurbi = false, avgRed = 0, avgGreen = 0, avgBlue = 0, dataTurbiRed = [], dataTurbiGreen = [], dataTurbiBlue = [];
        if (counter3 == 4) {
            isTurbi = true;
            counter3 = 0;
            avgRed = 1 * Math.random();
            avgGreen = 1 * Math.random();
            avgBlue = 1 * Math.random();
            for (var i = 0; i <= 15; i++) {
                var dataRed = { x: 2 * i, y: 100 * Math.random() };
                dataTurbiRed.push(dataRed);
                var dataGreen = { x: 2 * i, y: 100 * Math.random() };
                dataTurbiGreen.push(dataGreen);
                var dataBlue = { x: 2 * i, y: 100 * Math.random() };
                dataTurbiBlue.push(dataBlue);
            }
        }
        var dataMagn = [], avgMagn = 1 * Math.random();
        for (var i = 0; i <= 15; i++) {
            var obj = { x: 2 * i, y: 5 * Math.random() };
            dataMagn.push(obj);
        }

        couchBubblot1.insert("bubblot", {
            "data_key": [year, 06, 20, hours, minutes, seconds, 1], //change for each bubblot
            "data": {
                pumpLatitude: $scope.leftDataPump.localLat,
                pumpLongitude: $scope.leftDataPump.localLong,
                pumpOrientation: $scope.leftDataPump.gpsCompass,
                distancexToPump: $scope.rightData.distancexToPump,
                distanceyToPump: $scope.rightData.distanceyToPump,
                depth: $scope.rightData.depth * 100,
                temperature: 24,
                VA: {
                    available: isVa,
                    data: dataVa,
                    fe: fe,
                    pb: pb,
                    cu: cu,
                    sn: sn,
                },
                magnetism: {
                    data: dataMagn,
                    mean: avgMagn,
                },
                turbidity: {
                    available: isTurbi,
                    dataRed: dataTurbiRed,
                    dataGreen: dataTurbiGreen,
                    dataBlue: dataTurbiBlue,
                    avgRed: avgRed,
                    avgGreen: avgGreen,
                    avgBlue: avgBlue,
                },
                movie: {
                    available: isMovie,
                    data: [],
                }
            }
        }).then(({ data, headers, status }) => {
            // data is json response 
            // headers is an object with all response headers 
            // status is statusCode number 
        }, err => {
            // either request error occured 
            // ...or err.code=EDOCCONFLICT if document with the same id already exists 
        });
    }
    init();
}
])
