
(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('mapMap', mapMapDirective);

    function mapMapDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/map-map/map-map.html',
            controller: 'mapMapCtrl',
            controllerAs: 'vm',
            scope: {extractingData: '=', displayPath: '=', displayVa: '=', displayTurbi: '=', displayMovie: '=',displayFe: '=', displayPb: '=', displayCu: '=', displaySn: '=', isVa: '=', isFe: '=', isPb: '=', 
                    infoVa: '=', isCu: '=', isSn: '=', avgMagnetism: '=', infoMagnetism: '=', dataxPump:'=', datayPump: '=', datax:'=', datay: '=' ,  displayMagn: '=', 
                    isTurbi: '=', isMovie: '=', zoom: '=', ximage: '=', yimage: '=', xnew: '=', ynew: '=', vaCursor: '=', turbiCursor: '=', magnCursor: '=', movieCursor: '=', avgTurbi: '=', updatePanel:'=',
                    dates: '=', dateCursor:'=', latitudeCursor: '=', longitudeCursor:'=', depthCursor:'=', tempCursor:'=', infoTurbiRed: '=', infoTurbiGreen: '=', infoTurbiBlue: '=', xlast:'=', ylast:'=', 
                    bubblotCursor: '=', bubblots: '=', playData: '='},
            link: function(scope, element, attr) {
                scope.$watch('displayFe', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });    
                scope.$watch('displayPb', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayCu', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displaySn', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayPath', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayVa', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayMovie', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayMagn', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('displayTurbi', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('zoom', function () {
                    if(!scope.playData) scope.drawData(element.find('canvas')[0]);
                });
                scope.$watch('extractingData', function () {
                    scope.playTime = Math.max(scope.datax[0].length, scope.datax[1].length, scope.datax[2].length)-1;
                });
                scope.$watch('playData', function () {
                    if(scope.playData == true && scope.playTime == Math.max(scope.datax[0].length, scope.datax[1].length, scope.datax[2].length)-1){
                        scope.playTime=0;
                    } 
                    else if(scope.playData) scope.playTime++;
                });
                scope.$watch('playTime', function () {
                    scope.drawData(element.find('canvas')[0]);
                });
            }
        } 
    }
}());
