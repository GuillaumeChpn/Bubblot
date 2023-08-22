(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('winderAlertCtrl', winderAlertCtrl);

    function winderAlertCtrl($scope,$element) {
        var vm = this;
        var is1=false, is2=false, is3=false, is4=false;
        $scope.alert  = function(){
            var winder1=document.getElementsByClassName("alert-winder1"); 
            var winder2=document.getElementsByClassName("alert-winder2"); 
            var winder3=document.getElementsByClassName("alert-winder3"); 
            var winder4=document.getElementsByClassName("alert-winder4"); 
            var rail=document.getElementsByClassName("alert-rail"); 
            if(($scope.pressure1 && !$scope.reset1) || ($scope.pressure2 && !$scope.reset2) || 
                ($scope.pressure3 && !$scope.reset3) || ($scope.pressure4 && !$scope.reset4)){
                $scope.audio=true;
            }
            else{
                $scope.audio=false;
            }
            if(!($scope.pressure1 && !$scope.reset1)){
                for (var i = 0; i < winder1.length; i++) {
                    winder1[i].style.display='none';
                }
                is1=false;
            }
            if(!($scope.pressure2 && !$scope.reset2)){
                for (var i = 0; i < winder2.length; i++) {
                    winder2[i].style.display='none';
                }
                is2=false;
            }
            if(!($scope.pressure3 && !$scope.reset3)){
                for (var i = 0; i < winder3.length; i++) {
                    winder3[i].style.display='none';
                }
                is3=false;
            }
            if(!($scope.pressure4 && !$scope.reset4)){
                for (var i = 0; i < winder4.length; i++) {
                    winder4[i].style.display='none';
                }
                is4=false;
            }
            if(($scope.pressure1 && !$scope.reset1) && !is2 && !is3 && !is4){
                for (var i = 0; i < winder1.length; i++) {
                    winder1[i].style.display='block';
                }
                is1=true;
            }
            else if(($scope.pressure2 && !$scope.reset2) && !is1 && !is3 && !is4){
                for (var i = 0; i < winder2.length; i++) {
                    winder2[i].style.display='block';
                }
                is2=true;
            }
            else if(($scope.pressure3 && !$scope.reset3) && !is2 && !is1 && !is4){
                for (var i = 0; i < winder3.length; i++) {
                    winder3[i].style.display='block';
                }
                is3=true;
            }
            else if(($scope.pressure4 && !$scope.reset4) && !is2 && !is3 && !is1){
                for (var i = 0; i < winder4.length; i++) {
                    winder4[i].style.display='block';
                }
                is4=true;
            }
        }

    }

}());
