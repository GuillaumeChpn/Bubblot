(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('pumpAlertCtrl', ['$timeout', '$scope', pumpAlertCtrl]);

    function pumpAlertCtrl($timeout,$scope) {
        var vm = this;
        $scope.alert  = function(){
            var alertPump=document.getElementsByClassName("alert-pump"); 
            if($scope.security){
                $scope.audio=true;
                for (var i = 0; i < alertPump.length; i++) {
                    alertPump[i].style.display='block';
                }
            }
            else{
                $scope.audio=false;
                for (var i = 0; i < alertPump.length; i++) {
                    alertPump[i].style.display='none';
                }
            }
        }
    }

}());
