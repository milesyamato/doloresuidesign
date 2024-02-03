/**
 * Gulp plugin for include js/css/html... any file.
 * 
 * @version   v0.1.2
 * @author    Pandao <pandao@vip.qq.com>
 * @homePage  https://github.com/pandao/gulp-x-includer
 * @license   MIT license (MIT)
 * @copyright 2015 Pandao
 */ 

var fs      = require('fs');
var path    = require("path");
var util    = require('gulp-util');
var through = require("through2");

var PLUGIN_NAME = "gulp-x-includer";
      
function extend(dest, src, merge) {
    var keys = Object.keys(src), i = 0;

    while (i < keys.length) {
        if (!merge || (merge && dest[keys[i]] === undefined)) {
            dest[keys[i]] = src[keys[i]];
        }
        i++;
    }

    return dest;
}

module.exports = function (options) {
    options = options || {};
    
    if (typeof options.debug === "undefined") {
        options.debug = false;
    }
    
    if (typeof options.debugOptions === "undefined") {
        options.debugOptions = false;
    }
    
    var regexs = extend(options.regexs || {}, {
        js   : /\/\/\s*include\s*['"]([^'"]*)['"]\s*/g,
        html : /\<\!\-\-\s*include\s*['"]([^'"\>]*)['"]\s*\-\-\>/g,
        css  : /\/\*\s*include\s*['"]([^'"]*)['"]\s*\*\//g,
        file : /#\s*include\s*['"]([^'"]*)['"]\s*/g
    });

    var transform = options.transform || function(content) { return content;};
    
    if (options.debugOptions) {
        console.log(PLUGIN_NAME + ": options =>", regexs);
    }

    return through.obj(function (file, encoding, callback) {
        if (file.isNull()) {
            return callback(null, file);
        }

        if (file.isStream()) {
            this.emit('error', new util.PluginError(PLUGIN_NAME, 'Cannot use streamed files'));
            return callback();
        }

        if (file.isBuffer()) {
            var contents = file.contents.toString(encoding), matches;

            if (options.debug) {
                console.log(PLUGIN_NAME + ": target =>", file.path);
            }
            
            for (var i in regexs) {
                var regex = regexs[i];

                while (matches = regex.exec(contents)) {
                    var filePath = path.join(file.base, matches[1]);

                    if (options.debug) {
                        console.log(PLUGIN_NAME + ": include =>", filePath);
                    }

                    if (fs.existsSync(filePath)) {
                        var includeContents = fs.readFileSync(filePath, {encoding: encoding});

                        contents = contents.substr(0, matches.index) +
                                   transform(includeContents) +
                                   contents.substr(matches.index + matches[0].length);
                    } else {
                        this.emit('error', new util.PluginError(PLUGIN_NAME, "File not found: " + filePath));
                        return callback();
                    }
                }

                file.contents = new Buffer(contents);
            }

            if (options.debug) {
                console.log("\n");
            }
        }
        
        callback(null, file);
    });
};
