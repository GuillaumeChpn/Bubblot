(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('bubblotAlertCtrl', ['$timeout', '$scope', bubblotAlertCtrl]);

    function bubblotAlertCtrl($timeout,$scope) {
        var vm = this;
        $scope.alert  = function(){
            var security=document.getElementsByClassName("alert-security-bubblot")
            var twistLeftAlert=document.getElementsByClassName("alert-twist-left"); 
            var twistRightAlert=document.getElementsByClassName("alert-twist-right"); 
            if($scope.security || $scope.twistLeft || $scope.twistRight){
                $scope.audio=true;
            }
            else{
                $scope.audio=false;
            }
            if($scope.security && !$scope.twistLeft && !$scope.twistRight){
                for (var i = 0; i < security.length; i++) {
                    security[i].style.display='block';
                }
            }
            else if($scope.twistLeft && !$scope.security){
                for (var i = 0; i < twistLeftAlert.length; i++) {
                    twistLeftAlert[i].style.display='block';
                }
            }
            else if($scope.twistRight && !$scope.security){
                for (var i = 0; i < twistRightAlert.length; i++) {
                    twistRightAlert[i].style.display='block';
                }
            }
            if(!$scope.security){
                for (var i = 0; i < security.length; i++) {
                    security[i].style.display='none';
                }
            }
            if(!$scope.twistLeft){
                for (var i = 0; i < twistLeftAlert.length; i++) {
                    twistLeftAlert[i].style.display='none';
                }
            }
            if(!$scope.twistRight){
                for (var i = 0; i < twistRightAlert.length; i++) {
                    twistRightAlert[i].style.display='none';
                }
            }
        }

    }

}());
