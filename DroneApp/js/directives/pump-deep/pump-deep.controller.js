(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('pumpDeepCtrl', pumpDeepCtrl);

    function pumpDeepCtrl($scope) {
        var vm = this;
        var allScreen=document.getElementById('pumpDisplay');
        allScreen.addEventListener("wheel", scrollBallast, false);
        function scrollBallast(event) {
            var clientRect = allScreen.getBoundingClientRect();
            var minX = clientRect.left + (clientRect.width/5);
            var maxX = clientRect.right - (clientRect.width/5);
            if(event.screenX > minX && event.screenX < maxX){
                if($scope.ballastState>=0 && $scope.ballastState<=100){
                    $scope.ballastState=$scope.ballastState+3*event.deltaY/200;
                    if($scope.ballastState>100){
                        $scope.ballastState=100;
                    }
                    else if($scope.ballastState<0){
                        $scope.ballastState=0;
                    }
                    $scope.$apply();
                }
            }
        }
        $scope.checkSecurity  = function () {
            if($scope.security>70 || $scope.security<30){
                $scope.securityAlert = true;
            }
            else{
                $scope.securityAlert = false;
            }
        }
    }

} ());
