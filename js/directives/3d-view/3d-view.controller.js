(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('3dViewCtlr', ThreeDViewCtlr);

    /**@ngInject*/
    function ThreeDViewCtlr($scope) {
        var vm = this;
        var cadran=document.getElementsByClassName('cadrans-container view');
        for(var i=0; i<cadran.length; i++){
            cadran[i].addEventListener("wheel", scrollView, false);
        }
        function scrollView(event) {
            if($scope.focusIndex==1){
                if($scope.angle>=30 && $scope.angle<=45){
                    $scope.angle=$scope.angle+event.deltaY/200;
                    if($scope.angle>45){
                        $scope.angle=45;
                    }
                    else if($scope.angle<30){
                        $scope.angle=30;
                    }
                    $scope.$apply();
                }
            }
        }
        $scope.cursorPosition  = function () {
            //return 54*(($scope.angle-30)/60)+10;
            return 70-35*Math.tan((90-$scope.angle)* Math.PI/180);
        }
    }

}());
