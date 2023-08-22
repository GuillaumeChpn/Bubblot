(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('winderPanel1Ctrl', winderPanel1Ctrl);

    function winderPanel1Ctrl($scope, $element) {
        var vm = this;
        var winderLength=document.getElementsByClassName("winder1");
        var minPressure=document.getElementsByClassName("min-pressure1"); 
        var maxValue=18, minValue=6;
        updateScope();
        for (var i = 0; i < minPressure.length; i++) {
            if(i==0){
                minPressure[i].addEventListener('keyup', updatePressureDown);
            }
            else{
                minPressure[i].addEventListener('keyup', updatePressureUp);
            }
        }
        $scope.updateLength  = function (value) {
            for (var i = 0; i < winderLength.length; i++) {
                winderLength[i].innerHTML=value.toFixed(1);
            }
        }
        $scope.checkPressure  = function (value) {
            if($scope.pressure < $scope.minPressure){
                $scope.pressureAlert=true;
            }
            else{
                $scope.pressureAlert=false;
                $scope.reset=false;
            }
        }
        function updatePressureUp(event) {
            if(event.keyCode==13){
                    $scope.minPressure=(minPressure[1].value-minValue)/(maxValue-minValue);
                    updateScope();
            }
            else{
                minPressure[0].value="";
            }
        }
        function updatePressureDown(event) {
            if(event.keyCode==13){
                    $scope.minPressure=(minPressure[0].value-minValue)/(maxValue-minValue);
                    updateScope();
            }
            else{
                minPressure[1].value="";
            }
        }
        function updateScope(){
            for (var i = 0; i < minPressure.length; i++) {
                minPressure[i].value=($scope.minPressure*(maxValue-minValue)+minValue).toFixed(1);
            }
        }

        //To test change of pressure in winder panel
        var cadranWinderPanel=document.getElementsByClassName('cadrans-winder-container winder-panel left-one');
        for(var i=0; i<cadranWinderPanel.length; i++){
            cadranWinderPanel[i].addEventListener("wheel", scrollThrust, false);
        }
        function scrollThrust(event) {
            if($scope.pressure>=0 && $scope.pressure<=1){
                $scope.pressure=$scope.pressure+event.deltaY/10000;
                if($scope.pressure>1){
                    $scope.pressure=1;
                }
                else if($scope.pressure<0){
                    $scope.pressure=0;
                }
                $scope.$apply();
            }  
        }
    }

}());
