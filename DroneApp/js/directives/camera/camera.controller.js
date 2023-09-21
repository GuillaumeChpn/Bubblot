(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('cameraCtrl', cameraCtrl);

    function cameraCtrl($scope) {
        var vm = this,
            video = document.getElementsByClassName('cameraGear'),
            vendorURL = window.URL || window.webkitURL,
            record, recordedVideo = "", recordedObject = "",
            date = new Date(), month, minutes, fileName;

        /*
        navigator.getMedia = navigator.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozWetUserMedia ||
            navigator.msGetUserMedia;
        */
        function successCallback(stream) {
            var options = {
                mimeType: 'video/webm\;codecs=h264',
                bitsPerSecond: 128000
            };
            video[0].src = vendorURL.createObjectURL(stream);
            video[1].src = vendorURL.createObjectURL(stream);
            record = recordRTC(stream, options);
        }

        function errorCallback(error) {
            console.log("No cam detected, maybe another application is using the device");
        }

        var mediaConstraints = {
            audio: false,
            video: true
        };


        navigator.getUserMedia(mediaConstraints, successCallback, errorCallback);

        $scope.rec = function () {
            if ($scope.cameraRecMenu == false) {
                navigator.getMedia(mediaConstraints, successCallback, errorCallback);
                $scope.cameraRecMenu = true;
            }
            else if ($scope.cameraRec == false) {
                record.clearRecordedData();
                $scope.cameraRec = true;
                record.startRecording();
            }
            else {
                $scope.cameraRec = false;
                record.stopRecording(function (videoURL) {
                    recordedObject = record;
                    recordedVideo = videoURL;
                });
                navigator.getMedia(mediaConstraints, successCallback, errorCallback);
            }
        }

        $scope.save = function () {
            if (recordedObject != "" && !$scope.cameraRec) {
                month = date.getMonth() + 1;
                minutes = date.getMinutes();
                if (minutes < 10) {
                    minutes = "0" + minutes;
                }
                fileName = date.getFullYear() + "/" + month + "/" + date.getDate() + "/" + date.getHours() + "h" + minutes;
                recordedObject.save(fileName);
            }
        }
        $scope.play = function () {
            if (recordedVideo != "" && !$scope.cameraRec) {
                if ($scope.cameraRecMenu == true) {
                    $scope.cameraPlay = true;
                    $scope.cameraRecMenu = false;
                    video[0].src = recordedVideo;
                    video[1].src = recordedVideo;
                }
                else if ($scope.cameraPlay == true) {
                    $scope.cameraPlay = false;
                    video[0].pause();
                    video[1].pause();
                }
                else {
                    $scope.cameraPlay = true;
                    video[0].play();
                    video[1].play();
                }
            }

        }
    }

}());
