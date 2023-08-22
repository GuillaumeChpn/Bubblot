(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('lightingCtrl', lightingCtrl);

    function lightingCtrl($timeout, $scope) {
        var vm = this;
        var cadran=document.getElementsByClassName('cadrans-container lighting');
        for(var i=0; i<cadran.length; i++){
            cadran[i].addEventListener("wheel", scrollLighting, false);
        }
        function scrollLighting(event) {
            if($scope.focusIndex==3){
                var clientRect = this.getBoundingClientRect();
                var centerY = clientRect.top + (clientRect.height/2);
                if(event.screenY < centerY && $scope.spotlightSwitchOn){
                    if($scope.spotlightIntensity>=0 && $scope.spotlightIntensity<=100){
                        $scope.spotlightIntensity=$scope.spotlightIntensity+event.deltaY/100;
                        if($scope.spotlightIntensity > 100){
                            $scope.spotlightIntensity=100;
                        }
                        else if($scope.spotlightIntensity < 0){
                            $scope.spotlightIntensity=0;
                        }
                    }
                    $scope.$apply();
                }
                else if($scope.foglightSwitchOn){
                    if($scope.foglightIntensity>=0 && $scope.foglightIntensity<=100){
                        $scope.foglightIntensity=$scope.foglightIntensity+event.deltaY/100;
                        if($scope.foglightIntensity > 100){
                            $scope.foglightIntensity=100;
                        }
                        else if($scope.foglightIntensity < 0){
                            $scope.foglightIntensity=0;
                        }
                    }
                    $scope.$apply();
                }
            }
        }
        $scope.spotlightSwitchOnFunction = function () {
            $scope.spotlightSwitchOn=!$scope.spotlightSwitchOn;
            $timeout(function(){
                $scope.$apply();
            });
        }
        $scope.foglightSwitchOnFunction = function () {
            $scope.foglightSwitchOn=!$scope.foglightSwitchOn;
            $timeout(function(){
                $scope.$apply();
            });
        }
    }

}());
