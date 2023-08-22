(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('graph', graphDirective);

    function graphDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/graph/graph.html',
            replace: true,
            controller: 'graphCtrl',
            scope: { data: '=', computeVa: '='} ,
            link: function(scope, element, attr) {
                //scope.drawGraph(element.find('canvas')[0], scope.data);

                /*scope.$watch('data', function (data) {
                    scope.drawGraph(element.find('canvas')[0], scope.data);
                });       */
                scope.$watch('computeVa', function (value) {
                    if(value){
                        //scope.updateGraph(element.find('canvas')[0]);
                        //element.find(".water-va")[0].addClass = ".water-va.in";
                        //console.log(element.find(".water-va.in"));
                        setTimeout(function(){ scope.computeVa=false }, 23000);
                    }
                });     
            }
        }
    }

}());
