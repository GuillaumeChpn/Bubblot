(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('pumpGpsCtrl', pumpGpsCtrl);

    function pumpGpsCtrl($scope) {
        var vm = this;
        var localLat=document.getElementsByClassName("local-lat");
        var localLong=document.getElementsByClassName("local-long");
        var targetLat=document.getElementsByClassName("target-lat");
        var targetLong=document.getElementsByClassName("target-long");
        updateScope();
        for (var i = 0; i < targetLat.length; i++) {
            if(i==0){
                targetLat[i].addEventListener('keyup', updateTargetDown);
                targetLong[i].addEventListener('keyup', updateTargetDown);
            }
            else{
                targetLat[i].addEventListener('keyup', updateTargetUp);
                targetLong[i].addEventListener('keyup', updateTargetUp);
            }
        }
        function updateTargetUp(event) {
            if(event.keyCode==13){
                    $scope.targetLat=targetLat[1].value;
                    $scope.targetLong=targetLong[1].value;
                    updateScope();
            }
            else{
                targetLat[0].value="";
                targetLong[0].value="";
            }
        }
        function updateTargetDown(event) {
            if(event.keyCode==13){
                    $scope.targetLat=targetLat[0].value;
                    $scope.targetLong=targetLong[0].value;
                    updateScope();
            }
            else{
                targetLat[1].value="";
                targetLong[1].value="";
            }
        }
        function updateScope(){
            for (var i = 0; i < localLat.length; i++) {
                localLat[i].innerHTML=$scope.localLat;
                localLong[i].innerHTML=$scope.localLong;
                targetLat[i].value=$scope.targetLat;
                targetLong[i].value=$scope.targetLong;
            }
        }
    }
}());
