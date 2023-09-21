(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('deepCtrl', deepCtrl);

    function deepCtrl($scope, $element) {
        var vm = this;
        $scope.updateSonarDistance = function(value){
        }
        $scope.checkSecurity  = function () {
            if($scope.security>90 || $scope.security<10){
                //$scope.securityAlert = true;
            }
            else{
                //$scope.securityAlert = false;
            }
        }

        //To simulate change of security
        /*
        var cadran=document.getElementsByClassName('cadrans-container deep');
        for(var i=0; i<cadran.length; i++){
            cadran[i].addEventListener("wheel", scrollSecurity, false);
        }
        */
        function scrollSecurity(event) {
            if($scope.focusIndex==2){
                if($scope.security>=0 && $scope.security<=100){
                    $scope.security=$scope.security+event.deltaY/100;
                    if($scope.security>100){
                        $scope.security=100;
                    }
                    else if($scope.security<0){
                        $scope.security=0;
                    }
                    $scope.$apply();
                }
            }
        }
    }

} ());
