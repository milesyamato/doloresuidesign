/*! * Name: TynApp
  * Version: v1.0.0
  * Author: ILIASH HOSAIN.
  * Copyright: Themeyn 2023.
  * URI: https://themeforest.net/user/themeyn */
let TynApp = (function (win, doc) {
    "use strict";
    let TynApp = { 
        meta: { 
            name: "TynApp", version: "1.0.0", author: "Themeyn" 
        }, 
        app: { 
            name: "ConnectMe", version: "1.0", 
        } 
    }

    function Ready(callback){
        doc.addEventListener('DOMContentLoaded', callback, false);
    }
    function Load(callback){
        win.addEventListener('load', callback, false);
    }
    
    function Resize(callback,selector){
        selector = (typeof selector === typeof undefined) ? window : selector;
        selector.addEventListener('resize', callback)
    }

    TynApp.Ready = Ready;
    TynApp.Load = Load;
    TynApp.Resize = Resize;

    return TynApp;
}(window, document));

TynApp = function (TynApp) {
    "use strict";

    //Global Properties
    TynApp.Body = document.querySelector('body');
    TynApp.Style = function(name,value){
        if(name[0]!='-') name = '--'+name //allow passing with or without --
        if(value) document.documentElement.style.setProperty(name, value)
        return getComputedStyle(document.documentElement).getPropertyValue(name);
    }

    TynApp.Custom = {}
    TynApp.BS = {}
    TynApp.Plugins = {}
    TynApp.Appbar= {}
    TynApp.Page ={
        Height : window.innerHeight,
        Width : window.innerWidth,
    }
    TynApp.Breakpoints = {
        sm : parseInt(TynApp.Style('--bs-breakpoint-sm')),
        md : parseInt(TynApp.Style('--bs-breakpoint-md')),
        lg : parseInt(TynApp.Style('--bs-breakpoint-lg')),
        xl : parseInt(TynApp.Style('--bs-breakpoint-xl')),
        xxl : parseInt(TynApp.Style('--bs-breakpoint-xxl')),
    }
    TynApp.Color = {
        blue : TynApp.Style('--bs-blue'),
    }
    TynApp.Dark = TynApp.Body.classList.contains('dark-mode')
    TynApp.isRTL = TynApp.Body.getAttribute('dir') === 'rtl' ? true : false
    TynApp.hasTouch = ("ontouchstart" in document.documentElement) ? true : false
    TynApp.onMobile = (navigator.userAgent.match(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Windows Phone|/i)) ? true : false

    TynApp.Update = function(){
        TynApp.Page ={
            Height : window.innerHeight,
            Width : window.innerWidth,
        }
    }
    //get parent Elements
    TynApp.getParents = function(el, selector, filter) {
        let parentSelector = (selector === undefined) ? document : document.querySelector(selector);
        var parents = [];
        var pNode = el.parentNode;
        while (pNode !== parentSelector) {
            var element = pNode;
            if(filter === undefined){
                parents.push(element); 
            }else{
                element.classList.contains(filter) && parents.push(element);
            }
            pNode = element.parentNode;
        }
        return parents;
    }
    TynApp.Load(TynApp.Update);
    TynApp.Resize(TynApp.Update);

    return TynApp;
}(TynApp);